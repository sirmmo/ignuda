# Running Ollama (llama.cpp / ggml) on ignuda — gap analysis & roadmap

**Goal:** run Ollama on **integrated** Intel GPUs (Gen9 / Gen11 / Xe-LP) through
ignuda, by having Ollama's bundled llama.cpp **CUDA** backend load ignuda's
`libcuda`/`libcudart`/`libcublas` instead of NVIDIA's.

This document is the concrete "what is missing" answer. It is grounded in two
inventories taken 2026-10-13: ignuda's implemented surface, and the ggml CUDA
backend at llama.cpp commit `1b70350` (`ggml/src/ggml-cuda/`). Numbers below
come from those sources.

> **Status: analysis, not yet built.** Nothing here has been implemented. The
> workstreams are tracked in Flowly.

## The one idea that makes this feasible on an iGPU

ggml's CUDA backend is enormous (189 `.cu` files, ~166 kernels, FP8/FP4, tensor
cores, flash-attention). You do **not** need most of it, because:

1. **Integrated Intel GPUs have no matrix/tensor hardware.** XMX/DPAS lives on
   Arc discrete and the Xe-HPG/HPC parts; Gen9/Gen11/Xe-LP have none. So the
   `mma.sync` / `wmma` / `cp.async` tensor paths — the single hardest thing to
   translate — are exactly the paths you want ggml **not** to take. On an iGPU
   they would be emulated and slow even if translated.

2. **ggml picks its kernel path at runtime from the reported compute
   capability**, with graceful fallbacks all the way down to a generic scalar
   path. `turing_mma_available()`, `fast_fp16_hardware_available()`,
   `ggml_cuda_dp4a()` etc. all branch on the device cc and fall back when it is
   low.

3. **ggml ships fatbins containing PTX, not only SASS** (`-virtual` arches:
   `50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;…`). There is **no
   NVRTC** anywhere. ignuda receives JIT-able PTX modules — exactly what its
   PTX→SPIR-V front end wants — and its PTX-only fatbin ingestion already works.

**Therefore the lever is `IGNUDA_CC`.** ignuda's fatbin loader picks the highest
embedded PTX whose virtual arch ≤ the reported cc. Report a **low** cc and you
get both (a) ignuda JITing the *generic* PTX variant (no `mma`/`cp.async` in it,
because `__CUDA_ARCH__` gated those out at nvcc time) and (b) ggml's host code
choosing its *fallback* kernels. Report ignuda's current default of **8.6** and
ignuda grabs the `80-virtual` PTX full of `mma.sync`/`cp.async` and fails. So
step zero is: **stop reporting 8.6.**

There is a precision trade-off in *which* low cc:

| Reported cc | ggml behaviour | ignuda cost |
|---|---|---|
| **5.0 / 5.2** (Maxwell) | no `dp4a`, no fast-fp16 intrinsics, no `mma`; quantized matmul uses the **scalar int8 fallback**; GEMM goes to cuBLAS | **simplest PTX** — essentially what ignuda already compiles, plus packed-fp16 |
| **6.1** (Pascal) | enables `__dp4a` quantized dot path | requires `dp4a` in the compiler |
| ≥ 7.5 | enables tensor-core paths | requires `mma`/`cp.async` — **don't** |

Start at **5.2**, get tokens out, then raise to 6.1 once `dp4a` lands (it's the
biggest single quantized-inference speedup and maps cleanly to Intel's integer
dot-product extension).

## What is actually missing (for the low-cc iGPU path)

### MUST-HAVE — nothing generates a token without these

1. **cuBLAS `cublasGemmEx` + batched/strided variants, with FP16.**
   ggml calls `cublasGemmEx`, `cublasGemmStridedBatchedEx` (7 sites),
   `cublasGemmBatchedEx`, with `CUDA_R_16F`/`CUDA_R_32F`/`CUDA_R_16BF` and
   `CUBLAS_COMPUTE_16F`/`CUBLAS_COMPUTE_32F`. ignuda's cuBLAS today is
   `Sgemm`/`Dgemm` (FP32/FP64) **only** — no `GemmEx`, no FP16, no batching, no
   cuBLASLt (ggml doesn't use cuBLASLt — good). This is the **#1 host-side gap**.
   Note the GEMM itself can accumulate in FP32; only the A/B operand load needs
   FP16. (`ggml-cuda.cu:1405-1625`.)

2. **FP16 in the PTX→SPIR-V compiler — packed `f16x2`.**
   ggml is saturated with `half2` (501 occurrences) and `__half2float` /
   `__float2half` / `__hadd2` / `__hmul2`. ignuda handles *scalar* `.f16`
   (`OpTypeFloat 16`) but **packed `.f16x2`/`.bf16x2` conversions explicitly
   error** (`translate.rs:1870`), and there is no packed-half arithmetic. Intel
   iGPUs support `cl_khr_fp16` natively, so this is a translator gap, not a
   hardware one. This is the **#1 compiler gap**. (bf16 is already done in
   software; keep that.)

3. **`cudaGetDeviceProperties` / `cudaDeviceProp`.**
   ggml calls it per device and reads `major`/`minor`, `warpSize`, `name`,
   `totalGlobalMem`, `sharedMemPerBlock`, `multiProcessorCount`, `integrated`,
   `cudaDevAttrCooperativeLaunch`, etc. to choose paths and launch geometry.
   ignuda has **no `cudaGetDeviceProperties` at all** — it must be added and the
   struct filled consistently with the (low) reported cc. (The field offsets
   must match NVIDIA's `cudaDeviceProp` ABI exactly.)

4. **A few more runtime entry points ggml uses:**
   `cudaStreamPerThread` (per-thread default stream, 17 refs),
   `cudaMemcpy2DAsync` (7), `cudaSetDeviceFlags`, `cudaDeviceGetAttribute` for
   the attrs above, `cudaLaunchKernelEx` (ignuda **has** this — verify the
   `cudaLaunchConfig_t`/attribute path is honoured or safely ignored). ignuda's
   `cuStreamWaitEvent` is a no-op (single in-order queue); that is
   **correct-but-serial** and fine for a first bring-up.

5. **Build/run configuration (no ignuda code — this is how you invoke it):**
   Ollama/llama.cpp must be built so ignuda sees simple PTX and no unsupported
   host deps:
   - `-DGGML_CUDA_NO_VMM=ON` — removes the **driver-API VMM pool**
     (`cuMemCreate`/`cuMemMap`/`cuMemAddressReserve`/`cuMemSetAccess`), which
     ignuda does not implement; falls back to `cudaMalloc`.
   - `-DGGML_CUDA_USE_GRAPHS=OFF` — ignuda has no `cudaGraph*`.
   - `-DGGML_CUDA_NO_PEER_COPY=ON` — single device, no P2P.
   - `-DCMAKE_CUDA_ARCHITECTURES=52-virtual` (or `52-virtual;61-virtual` once
     dp4a lands) — keep the embedded PTX generic and matched to `IGNUDA_CC`.
   - Consider `-DGGML_CUDA_FORCE_CUBLAS=ON` initially to route all matmuls
     through cuBLAS (fewer device kernels to get right) — but this makes
     requirement **#1** even more central.
   - Then at runtime: `IGNUDA_CC=5.2`, `IGNUDA_DEVICE_TYPE=gpu`,
     `LD_LIBRARY_PATH` pointing at ignuda's libs under NVIDIA's sonames.

### SHOULD-HAVE — correctness for more models, or meaningful speed

6. **`dp4a` (int8 byte dot-product)** in the compiler → unlocks reporting cc 6.1
   and ggml's fast quantized MMQ/MMVQ path (177 uses). Maps to Intel's
   `cl_khr_integer_dot_product` (`dot_4x8packed_*`) on Gen11+/Xe, or the
   `cl_intel`/DP4A path. Biggest quantized-inference win after the must-haves.

7. **Device-cap guards ignuda currently skips.** The compiler emits
   `OpTypeFloat 16` and subgroup ops but ignuda **does not check** the device
   advertises `cl_khr_fp16` or `cl_khr_subgroups`. Add the guards so a device
   that can't do it fails cleanly instead of miscompiling.

8. **`cudaStream` real concurrency** (currently one in-order queue). Correct but
   serial; revisit only if it's a measured bottleneck.

### DO NOT BUILD (for the iGPU target)

- `mma`/`wmma`/`ldmatrix`/`movmatrix`, `cp.async`, tensor-core paths — no iGPU
  hardware; keep ggml off them via low cc.
- FP8 (`__nv_fp8*`) and FP4/MXFP4 (`__nv_fp4*`, sm_120) — Blackwell-only.
- CUDA Graphs, mem pools (`cudaMemPool*` — ggml doesn't use them anyway),
  VMM driver pool, P2P/`cudaMemcpyPeer`, NCCL/`allreduce`, cooperative-groups
  grid sync (`cudaLaunchCooperativeKernel`, one kernel — let it fail or stub).
- Textures/surfaces, cuDNN, cuBLASLt (ggml uses none of these).

## iGPU-specific gotchas (these bite on hardware, not on the CPU runtime)

- **Warp size vs Intel subgroup size.** ggml hard-assumes `WARP_SIZE == 32` and
  reduces with `__shfl_xor_sync` (34 uses). ignuda emits a required
  `SubgroupSize = 32` execution mode. Intel EUs are natively SIMD8/16; **32 is
  not always an available subgroup size on Gen9/Gen11**. You must request it via
  `cl_intel_required_subgroup_size` (SIMD32 is supported on most Intel GPUs for
  compute, but confirm per device) or the kernel won't build. This is the most
  likely first on-hardware failure. Verify `CL_DEVICE_SUB_GROUP_SIZES_INTEL`
  includes 32.
- **USM on older Gen.** ignuda **requires** `cl_intel_unified_shared_memory`.
  Gen9/early drivers may expose only partial USM (host/device, not shared) — or
  none. Check `clinfo` for the extension before expecting anything.
- **No FP64 on Gen11.** Irrelevant to Ollama (models are fp16/fp32/quant); just
  don't let `Dgemm` or any `.f64` PTX reach the device.
- **Shared-memory ceiling.** Intel SLM is ~64KB/subslice; ggml requests dynamic
  shared memory via `cudaFuncAttributeMaxDynamicSharedMemorySize`. The low-cc
  tile/scalar kernels are modest, but watch for kernels that assume ≥48KB.
- **`-cl-fp32-correctly-rounded-divide-sqrt`** is already passed; good for
  matching CUDA numerics, but combined with ggml's `-use_fast_math` PTX
  (`sin`/`cos`/`ex2` → OpenCL `native_*`) expect small numeric drift. Validate
  output quality, not bit-exactness.

## Phased plan

- **Phase 0 — prove the loader path (no new ignuda features).** Build
  llama.cpp CUDA with the flags above and `52-virtual`; run the smallest model
  with `IGNUDA_CC=5.2 IGNUDA_DEVICE_TYPE=cpu` first (CPU OpenCL runtime, which
  is the only thing validated today). Expect it to fail at `cublasGemmEx` and/or
  packed-fp16 — that failure *is* the Phase-0 deliverable: it confirms the gap
  list and the loader wiring before touching an iGPU.
- **Phase 1 — must-haves.** `cublasGemmEx` (+ strided/batched) with FP16;
  packed `f16x2` in the compiler; `cudaGetDeviceProperties`; the handful of
  runtime entry points. Target: one small Q4/F16 model generating coherent
  tokens on the **CPU** runtime.
- **Phase 2 — first iGPU bring-up.** Move to a Gen11/Xe-LP iGPU. Resolve the
  subgroup-size-32 and USM-extension gotchas. This is where Flowly #454 (real
  iGPU testing) finally gets exercised with a real workload.
- **Phase 3 — speed.** `dp4a` + report cc 6.1; multi-stream concurrency;
  cuBLAS GEMM tiling (or dispatch GEMM to Intel oneMKL). Only after correctness.

## Phase 0 results (verified 2026-10-13)

Phase 0 was run against the freshly built libraries. **A full end-to-end Ollama
run could not be done here**: building ggml's CUDA device code needs `nvcc`
(the ignuda dev image is deliberately NVIDIA-free), so a real test needs either
nvcc-built fatbins or a prebuilt Ollama CUDA runner — that packaging is itself
Phase-1 infrastructure. Instead the gap list was confirmed directly against
ignuda's exported symbols and the compiler's actual output. Results:

**cuBLAS (`libcublas.so`)** — present: `cublasCreate_v2`, `cublasSetStream_v2`,
`cublasSetMathMode`, `cublasSetWorkspace_v2`, `cublasSgemm_v2`. **Missing (real
gaps):** `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`
(core inference), `cublasSgemmStridedBatched`, `cublasSgemmBatched`,
`cublasStrsmBatched` (the last only in `solve_tri` — non-core), and
`cublasGetStatusString` / `cublasLoggerConfigure` (error/logging — trivial
stubs).

**CUDA runtime (`libcudart.so`)** — present: `cudaDeviceGetAttribute`,
`cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaStreamCreateWithFlags`,
`cudaEventCreateWithFlags`, `cudaHostGetDevicePointer`, **`cudaLaunchKernelExC`**.
**Missing (real gaps):** `cudaGetDeviceProperties_v2` (the CUDA-12 symbol the
header `#define`s `cudaGetDeviceProperties` to — **this is the one that matters**),
`cudaSetDeviceFlags`, `cudaMemcpy2DAsync`, `cudaDeviceGetPCIBusId` (logging —
minor). The peer-access entry points (`cudaDeviceCanAccessPeer`,
`cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync`) and all VMM driver symbols
(`cuMemCreate`/`cuMemMap`/…) are also absent, but those are avoided by the
`-DGGML_CUDA_NO_PEER_COPY` / `-DGGML_CUDA_NO_VMM` build flags, so they are not
gaps for this path.

**Correction to the must-have list:** `cudaLaunchKernelEx` is **not** a gap —
it is a header-inlined C++ wrapper over `cudaLaunchKernelExC`, which ignuda
already exports. Requirement #4 above is narrowed accordingly.

**Compiler** — the packed-fp16 gap reproduces exactly. A kernel with
`cvt.rn.f16x2.f32` fails to compile with:

```
error: in function pack: line 13: cvt.rn.f16x2.f32: packed half conversions are not supported yet
```

So the Phase-1 must-haves are confirmed and sharpened: **`cublasGemmEx` +
strided/batched (FP16)**, **packed `f16x2` in the compiler**, and
**`cudaGetDeviceProperties_v2`** (plus the small trio `cudaSetDeviceFlags` /
`cudaMemcpy2DAsync` / `cudaDeviceGetPCIBusId`). Everything else ggml needs is
already exported or is avoided by build flags.

### Phase-1 progress

- **Packed `f16x2`/`bf16x2` in the compiler — done** (`cvt`/add/sub/mul/fma/
  min/max/neg/abs), verified numerically on the CPU runtime. See
  `crates/ptx2spirv/tests/packed_half.rs`.
- **`cublasGemmEx` + `cublasGemmStridedBatchedEx` + `cublasGemmBatchedEx`
  (FP16) — done.** A hand-written FP16 GEMM (`kernels/gemm_half.ptx`, since the
  dev image has no nvcc) covers FP16 A/B with FP16 or FP32 C; the all-FP32 case
  reuses `sgemm`; unsupported type combinations return `CUBLAS_STATUS_NOT_
  SUPPORTED`. alpha/beta are read as `__half` or `float` per the compute type.
  Verified end to end against a CPU reference in `samples/cublas_gemmex_test.c`.
  Not yet covered: BF16 inputs (`CUDA_R_16BF`) and INT8 (`dp4a`) — follow-ups.
- **`cudaGetDeviceProperties_v2` — pending.** Needs the exact `cudaDeviceProp`
  struct layout, which must be pinned against the real CUDA 12 header.

## Reality check

The host-API surface is ~90% there already; the real work is **two focused
pieces** (FP16 `cublasGemmEx`, and packed-fp16 in the compiler) plus
`cudaGetDeviceProperties`, and then the **iGPU hardware gotchas** (subgroup
size, USM) that can only be found on real silicon. This is weeks of focused
work, not a rewrite — *provided* you hold ggml to its low-cc fallback paths and
resist the temptation to translate tensor-core PTX that an integrated GPU can't
accelerate anyway. Ollama itself needs no patching; it's the llama.cpp CUDA
runner build flags and the ignuda gaps above.

Cross-references: `docs/ARCHITECTURE.md` (how PTX reaches the GPU),
`README.md` (env vars, limitations), `RELEASING.md`.
