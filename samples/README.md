# Samples

Small end-to-end checks for the three ignuda layers. Build them against ignuda
with `../run-tests.sh`, or by hand with the libraries on `LD_LIBRARY_PATH`.

- `kernels.cu` / `kernels.ptx` — kernels used by the driver-API test
  (`kernels.ptx` is `kernels.cu` compiled with `nvcc -ptx` for convenience).
- `driver_test.c` — loads `kernels.ptx` through the **driver API** and checks
  vadd, saxpy, a shared-memory reduction with atomics, and a tiled matmul.
- `runtime_test.cu` — an ordinary **runtime-API** program (built with nvcc)
  whose kernel runs through ignuda's `libcudart`.
- `cublas_test.c` — calls `cublasSgemm` and compares against a CPU reference.

The compiled binaries are intentionally not checked in.

`driver_test` and `cublas_test` need no CUDA toolkit: `driver_test` builds
against the bundled clean-room `include/cuda.h` (a minimal, hand-written subset
of the public driver API — not NVIDIA's header), and `cublas_test` declares the
few prototypes it uses itself. `runtime_test` is the only sample that needs
`nvcc` (any CUDA toolkit, no GPU required), and must be built with
`-cudart shared` so it loads ignuda's runtime rather than statically linking
NVIDIA's:

```sh
gcc -Iinclude -o driver_test driver_test.c -lcuda
gcc -o cublas_test cublas_test.c -lcudart -lcublas
# runtime_test (needs nvcc):
nvcc -ptx -arch=compute_61 -o kernels.ptx kernels.cu   # regenerate kernels.ptx
nvcc -o runtime_test runtime_test.cu -cudart shared -gencode arch=compute_61,code=compute_61
```
