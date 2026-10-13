# Architecture

ignuda is a stack of clean-room replacements for NVIDIA's user-mode CUDA
libraries. Each layer below depends only on the layer under it, and the bottom
layer targets OpenCL. This document explains how a CUDA program reaches the
Intel GPU and why each piece exists.

## The layers

### `crates/ptx` — PTX front end

A lexer and parser for NVIDIA's PTX virtual ISA. The grammar is deliberately
shallow: instructions keep their opcode and dot-modifiers as strings
(`add.f32.rn` → op `add`, mods `["f32","rn"]`), and the compiler interprets
them. An unsupported instruction then becomes a clear compile-time error
naming the opcode, rather than a parse failure. The parser tolerates the
metadata nvcc emits (`.loc`, `.file`, `.pragma`, `.callprototype`, linkage
directives, parameterised register banks like `%r<5>`).

### `crates/ptx2spirv` — PTX → SPIR-V compiler

Translates a parsed PTX module into a SPIR-V module for the OpenCL **Kernel**
execution model (the environment Intel's compute runtime accepts). Key
decisions:

- **Registers become `Function`-storage variables.** Each PTX register is an
  `OpVariable`; reads and writes are loads and stores. The device compiler's
  own mem2reg/SROA turns these back into SSA values, so we get correct
  semantics without building SSA ourselves.
- **Device pointers are integers.** PTX has a flat, untyped address space;
  SPIR-V is typed. We keep addresses as `i64` and materialise a typed pointer
  (`OpConvertUToPtr` / `OpBitcast`) at each memory access. This matches how CUDA
  code does pointer arithmetic.
- **Shared memory is anchor-relative.** `Workgroup` storage has no integer
  address, but PTX freely stores shared addresses in 32-bit registers. We keep
  a per-module `Workgroup` anchor variable and represent a shared address as a
  byte offset from it, converting to/from a real `Workgroup` pointer at access
  time.
- **Control flow is unstructured.** PTX branches become `OpBranch` /
  `OpBranchConditional` between labelled blocks. The Kernel capability permits
  unstructured control flow, so no reducibility analysis is needed.
- **Builtins** (`%tid`, `%ntid`, …) map to SPIR-V `BuiltIn` decorated
  variables; warp ops map to the `GroupNonUniform*` family with a required
  subgroup size.

The compiler reports, per kernel, the parameter layout (so the driver can set
OpenCL kernel arguments with the right sizes), the static shared-memory size,
and whether dynamic shared memory is used. It also synthesises a helper kernel
that writes the device addresses of module-scope globals, which is how
`cuModuleGetGlobal` is implemented.

### `crates/cuda` — `libcuda.so.1` (driver API)

The driver API on top of OpenCL + the `cl_intel_unified_shared_memory` (USM)
extension.

- **Devices** are OpenCL devices that advertise both SPIR-V and USM. Each gets
  one OpenCL context; contexts own an in-order command queue.
- **Memory** is USM: device/host/shared allocations map onto the
  `clDeviceMemAllocINTEL` / `clHostMemAllocINTEL` / `clSharedMemAllocINTEL`
  family. Because USM pointers are ordinary addresses, `CUdeviceptr` is just
  that address and host/device copies are `clEnqueueMemcpyINTEL`.
- **Modules** JIT the PTX (or the PTX extracted from a fatbin) to SPIR-V,
  build it with `clCreateProgramWithIL` + `clBuildProgram`, and wrap each
  kernel.
- **Launch** translates a CUDA grid/block into an OpenCL global/local
  ND-range (global size = grid × block) and forwards kernel arguments.

Two pieces of NVIDIA's *undocumented* surface are reimplemented because the
CUDA runtime calls them before it will do anything: the **fatbin** container
format (including its LZ4-compressed entries) and the `cuGetExportTable`
tables. These are clean-room implementations of public-effect interfaces; they
contain no NVIDIA code.

### `crates/cudart` — `libcudart.so.12` (runtime API)

The layer an nvcc-compiled program's host code actually calls. nvcc emits, per
translation unit, registration calls that this library implements:

- `__cudaRegisterFatBinary` stashes the embedded device image;
- `__cudaRegisterFunction` ties each `__global__` stub's host address to a
  device kernel name;
- `__cudaRegisterVar` does the same for `__device__`/`__constant__` globals;
- a `kernel<<<grid,block>>>` launch lowers to
  `__cudaPushCallConfiguration` + a call to the stub, which calls
  `cudaLaunchKernel` with its own address — which we resolve back to the
  registered kernel.

Everything else (`cudaMalloc`, `cudaMemcpy`, streams, events, device
attributes) forwards to the driver API, which this library loads from
`libcuda` at runtime so that the application and the runtime share one driver
state. Because ignuda provides its own runtime, **NVIDIA's runtime is never
loaded**, and the integrity check inside it never runs.

### `crates/cublas` — `libcublas.so.12`

A minimal BLAS. The kernels are compiled from `kernels/gemm.cu` to PTX once
(checked in as `kernels/gemm.ptx`) and embedded; at runtime they are JIT'd
through the driver like any other module. Correctness first: the GEMM is a
one-thread-per-output-element kernel honouring column-major layout, leading
dimensions, transpose flags, and host/device scalar pointer modes.

## Data flow of one kernel launch (runtime API)

1. At load time, the C runtime calls `__cudaRegisterFatBinary` →
   `__cudaRegisterFunction` for each kernel. ignuda records the image and the
   stub→name map; nothing touches the device yet.
2. The app calls `kernel<<<g,b>>>(args)`. nvcc's glue calls
   `__cudaPushCallConfiguration(g,b,…)` then the host stub, which calls
   `cudaLaunchKernel(&stub, …)`.
3. First use binds the device's primary context and JITs every registered
   module (PTX→SPIR-V→`clBuildProgram`).
4. `cudaLaunchKernel` looks up the `CUfunction` for `&stub`, then calls
   `cuLaunchKernel`, which sets OpenCL kernel args and enqueues the ND-range.

## Portability

The code is written to build for both Linux (`libcuda.so.1` / `libcudart.so.12`)
and Windows (`nvcuda.dll` / `cudart64_12.dll`); the OpenCL and dynamic-loader
shims pick the right names per platform. Only Linux has been built and tested
so far.
