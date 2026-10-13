# ignuda

[![CI](https://github.com/sirmmo/ignuda/actions/workflows/ci.yml/badge.svg)](https://github.com/sirmmo/ignuda/actions/workflows/ci.yml)

Run CUDA programs on Intel GPUs — a drop-in `libcuda` / `libcudart` that
compiles NVIDIA PTX to SPIR-V and executes it through OpenCL + Intel's unified
shared memory, in the spirit of [ZLUDA](https://github.com/vosen/ZLUDA).

A CUDA application loads ignuda's libraries instead of NVIDIA's and runs
unmodified. No NVIDIA driver, no NVIDIA runtime, and no NVIDIA GPU are
involved at any point.

> **Status: early prototype.** The driver API, the runtime API, and a minimal
> cuBLAS all work for the sample programs below. It has so far been exercised
> only on Intel's OpenCL **CPU** runtime (the development machine has no Intel
> GPU); running on a real Gen9/Gen11 iGPU is the next milestone. Expect gaps.

## What works today

- **PTX → SPIR-V compiler** (`crates/ptx`, `crates/ptx2spirv`): a PTX parser and
  a SPIR-V (OpenCL kernel environment) code generator covering the common
  integer/float/control-flow/memory instructions, shared memory, atomics
  (including a CAS fallback for float min/max and `inc`/`dec`), and warp-level
  `shfl`/`vote`/`ballot`. Output passes `spirv-val`.
- **`libcuda.so.1`** (`crates/cuda`): ~100 driver API entry points — contexts,
  modules, memory (via USM), streams, events, and kernel launch — plus the
  compressed **fatbin** parser that reads the device code nvcc embeds in host
  binaries, and the undocumented `cuGetExportTable` tables the CUDA runtime
  expects.
- **`libcudart.so.12`** (`crates/cudart`): the CUDA runtime API and the nvcc
  host-side registration glue (`__cudaRegisterFatBinary`,
  `__cudaRegisterFunction`, `__cudaPushCallConfiguration`, …). This is ignuda's
  own runtime, built directly on the driver layer — NVIDIA's runtime is never
  loaded.
- **`libcublas.so.12`** (`crates/cublas`): a minimal, correctness-first cuBLAS
  (`Sgemm`/`Dgemm`/`Saxpy`/`Sscal`) implemented with PTX kernels JIT'd through
  the driver. Performance work (tiling, or dispatching to Intel oneMKL) is
  future work.

The sample suite (`samples/`) covers all three layers: a driver-API program
that loads hand-written PTX, an nvcc-compiled runtime-API program, and a
cuBLAS SGEMM checked against a CPU reference. All pass.

## How it fits together

```
   CUDA application (unmodified, built by nvcc)
        │  cudaMalloc, kernel<<<>>>, cublasSgemm, …
        ▼
   libcublas.so.12   libcudart.so.12     ← ignuda (this repo)
        └──────┬──────────┘
               ▼
          libcuda.so.1                   ← ignuda: driver API + PTX→SPIR-V JIT
               ▼
   OpenCL ICD loader  +  Intel USM extension
               ▼
   Intel compute runtime  (iGPU, or the OpenCL CPU runtime)
```

PTX is JIT-compiled to SPIR-V the first time a module is loaded. Device
pointers are plain 64-bit integers backed by USM allocations, mirroring CUDA's
flat address space, so host and device share one address space exactly as
unified memory does on an integrated GPU.

## Requirements

- An OpenCL 3.0 device exposing `cl_khr_il_program` (SPIR-V) and
  `cl_intel_unified_shared_memory`. On Intel GPUs that is the
  [compute runtime](https://github.com/intel/compute-runtime); for development
  without a GPU, Intel's OpenCL CPU runtime works.
- The OpenCL ICD loader (`libOpenCL.so.1` / `OpenCL.dll`).
- A recent stable Rust toolchain.

## Build

```sh
cargo build --release
```

This produces `target/release/`:

| file                  | install as          |
|-----------------------|---------------------|
| `libcuda.so`          | `libcuda.so.1`      |
| `libcudart.so`        | `libcudart.so.12`   |
| `libcublas.so`        | `libcublas.so.12`   |

To run a CUDA program against ignuda, put these (named as in the right-hand
column) on a directory and point `LD_LIBRARY_PATH` at it ahead of any NVIDIA
install.

A containerised toolchain (Rust + Intel OpenCL CPU runtime + SPIR-V tools) is
provided for reproducible builds and testing:

```sh
docker build -t ignuda-dev -f docker/dev.Dockerfile docker/
./run-tests.sh         # builds and runs the sample suite in the container
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, using no
NVIDIA software at all:

- **build** — builds every crate for Linux, cross-compiles the three libraries
  for Windows (`x86_64-pc-windows-gnu`), runs the tests, then compiles the
  committed PTX to SPIR-V and validates it with `spirv-val`. The resulting
  Linux `.so` / Windows `.dll` libraries are uploaded as a build artifact.
- **e2e** — installs Intel's OpenCL CPU runtime and runs the driver-API and
  cuBLAS samples against ignuda end to end (the samples build against a bundled
  clean-room `cuda.h`, so no CUDA toolkit is needed).
- **lint** — `rustfmt` / `clippy`, advisory for now.

## Environment variables

| variable              | effect |
|-----------------------|--------|
| `IGNUDA_LOG=1`        | log every driver call and compilation |
| `IGNUDA_DEVICE_TYPE`  | `gpu` \| `cpu` \| `all` — which OpenCL devices to expose (default: GPU, else all) |
| `IGNUDA_CC=8.6`       | compute capability reported to the application |
| `IGNUDA_OPENCL=/path` | explicit OpenCL ICD loader to load |
| `IGNUDA_DUMP=/dir`    | dump every PTX module and the SPIR-V it compiled to |
| `IGNUDA_SYNC=1`       | synchronise after each launch (debugging) |

## Scope and limitations

- **Tested on the OpenCL CPU runtime only** so far; a Gen9/Gen11 iGPU is the
  next target. Gen11 has no native fp64, which will need emulation.
- Apps that **statically** link NVIDIA's `libcudart` (nvcc's default) must be
  rebuilt with `-cudart shared` to pick up ignuda's runtime. Shared linking is
  what PyTorch and most distributed binaries already use.
- cuBLAS is correctness-first, not yet fast; cuDNN is not started.
- Textures/surfaces, CUDA graphs, cooperative groups, MPS, and IPC are not
  implemented.
- `mma`/`wmma` tensor-core instructions and `cp.async` (used by recent
  architectures' PTX) are not compiled yet.

## A note on compatibility

ignuda does not bypass, disable, or defeat any licensing or integrity check in
NVIDIA's software. It works by *replacing* the NVIDIA user-mode libraries
entirely with a clean-room implementation, so those libraries — and any checks
inside them — are simply never loaded. The design deliberately avoids loading
any NVIDIA component.

## Licence

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), or
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option — the same terms as [ZLUDA](https://github.com/vosen/ZLUDA).
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work shall be dual-licensed as above, without
any additional terms or conditions.

## Credits

Inspired by Andrzej Janik's [ZLUDA](https://github.com/vosen/ZLUDA). The PTX
semantics follow NVIDIA's publicly documented
[Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/);
the SPIR-V target follows the Khronos
[SPIR-V](https://registry.khronos.org/SPIR-V/) and OpenCL environment specs.
