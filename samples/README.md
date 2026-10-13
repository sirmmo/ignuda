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

The compiled binaries are intentionally not checked in. `nvcc` (any CUDA
toolkit, no GPU required) builds the `.cu` sources; `runtime_test` must be
built with `-cudart shared` so it loads ignuda's runtime:

```sh
nvcc -ptx -arch=compute_61 -o kernels.ptx kernels.cu
nvcc -o runtime_test runtime_test.cu -cudart shared -gencode arch=compute_61,code=compute_61
gcc -o driver_test driver_test.c -lcuda
gcc -o cublas_test cublas_test.c -lcudart -lcublas
```
