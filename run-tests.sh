#!/bin/sh
# Build ignuda and run the sample test suite against it, with no NVIDIA
# libraries on the path. Runs inside the ignuda-dev container via dev.sh.
set -e
./dev.sh '
  set -e
  cargo build -p cuda -p cudart -p cublas
  LIB=/tmp/ignuda-lib
  rm -rf $LIB && mkdir -p $LIB
  cp target/debug/libcuda.so   $LIB/libcuda.so.1
  cp target/debug/libcudart.so $LIB/libcudart.so.12
  cp target/debug/libcublas.so $LIB/libcublas.so.12
  ln -sf libcudart.so.12 $LIB/libcudart.so
  ln -sf libcublas.so.12 $LIB/libcublas.so
  cd samples
  echo "== driver API (hand-written PTX) =="
  LD_LIBRARY_PATH=$LIB ./driver_test kernels.ptx
  echo "== runtime API (nvcc fatbin) =="
  LD_LIBRARY_PATH=$LIB ./runtime_test
  echo "== cuBLAS SGEMM =="
  gcc -o cublas_test cublas_test.c -L$LIB -lcudart -lcublas
  LD_LIBRARY_PATH=$LIB ./cublas_test
'
