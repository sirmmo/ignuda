# Status and roadmap

_Last updated 2026-10-13._

## Verified working

| area | state | tested on |
|------|-------|-----------|
| PTX → SPIR-V (common ISA) | working, `spirv-val` clean | — |
| Driver API (`libcuda`)    | vadd, saxpy, shared-mem reduction + atomics, tiled matmul | Intel OpenCL CPU runtime |
| Runtime API (`libcudart`) | nvcc-compiled program runs unmodified | Intel OpenCL CPU runtime |
| cuBLAS SGEMM/DGEMM         | exact match vs CPU reference | Intel OpenCL CPU runtime |
| fatbin (LZ4) extraction    | working | — |
| `cuGetExportTable` glue    | runtime initialises | — |

## Not yet done

- **Run on a real Intel iGPU** (Gen9/Gen11). Everything so far is on the CPU
  OpenCL runtime. _(Flowly #454)_
- **cuBLAS performance** — currently one thread per output element. Options:
  tiled kernels, or dispatch to Intel oneMKL.
- **cuDNN** (`libcudnn`) — not started; needed for most of PyTorch's NN ops.
- **Compiler gaps** for recent PTX: `mma`/`wmma` (tensor cores), `cp.async`,
  packed fp16x2/bf16x2 conversions, `prmt` non-default modes, `brx.idx`.
- **fp64 emulation** for Gen11 (no native double).
- **Windows build** (`nvcuda.dll` / `cudart64_12.dll`) — code is portable but
  unbuilt.
- **Device `printf`** (`vprintf`) and `assert`.
- Textures/surfaces, CUDA graphs, cooperative launch, IPC, MPS.

## Milestones

1. ✅ Driver API + JIT: hand-written and nvcc PTX run.
2. ✅ Runtime API: nvcc programs run without NVIDIA's runtime.
3. ✅ Minimal cuBLAS.
4. ⏳ Validate on a physical Gen9/Gen11 iGPU.
5. ⏳ A real PyTorch tensor op end-to-end (needs cuBLAS speed + partial cuDNN).
6. ⏳ Windows.

## Known design limits

- Statically linked NVIDIA cudart needs rebuilding with `-cudart shared`.
- One in-order queue per context: streams serialise rather than overlap.
