#!/bin/sh
# Build ignuda and run the sample test suite against it, with no NVIDIA
# libraries on the path. Runs inside the ignuda-dev container via dev.sh.
#
# driver_test and cublas_test need no NVIDIA toolkit (the samples ship a
# clean-room cuda.h); runtime_test is built and run only when nvcc is present.
set -e
./dev.sh '
  set -e
  cargo build -p cuda -p cudart -p cublas
  LIB=/tmp/ignuda-lib
  rm -rf $LIB && mkdir -p $LIB
  cp target/debug/libcuda.so   $LIB/libcuda.so.1
  cp target/debug/libcudart.so $LIB/libcudart.so.12
  cp target/debug/libcublas.so $LIB/libcublas.so.12
  ln -sf libcuda.so.1   $LIB/libcuda.so
  ln -sf libcudart.so.12 $LIB/libcudart.so
  ln -sf libcublas.so.12 $LIB/libcublas.so
  cd samples
  export LD_LIBRARY_PATH=$LIB

  echo "== driver API (hand-written PTX) =="
  gcc -Iinclude -o driver_test driver_test.c -L$LIB -lcuda
  ./driver_test kernels.ptx

  echo "== cuBLAS SGEMM =="
  gcc -o cublas_test cublas_test.c -L$LIB -lcudart -lcublas
  ./cublas_test

  echo "== cuBLAS GemmEx (FP16) =="
  gcc -o cublas_gemmex_test cublas_gemmex_test.c -L$LIB -lcudart -lcublas -lm
  ./cublas_gemmex_test

  echo "== cudaGetDeviceProperties =="
  gcc -o device_props_test device_props_test.c -L$LIB -lcudart
  IGNUDA_CC=5.2 ./device_props_test

  if command -v nvcc >/dev/null 2>&1; then
    echo "== runtime API (nvcc fatbin) =="
    nvcc -o runtime_test runtime_test.cu -cudart shared -gencode arch=compute_61,code=compute_61
    ./runtime_test
  else
    echo "== runtime API: skipped (nvcc not installed) =="
  fi
'
