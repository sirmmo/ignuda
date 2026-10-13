# Development image: Rust toolchain + Intel OpenCL CPU runtime (accepts SPIR-V,
# shares the SPIR-V front end with the iGPU driver) + SPIR-V tools for validation.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg build-essential pkg-config git \
      ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers clinfo spirv-tools \
      gcc-mingw-w64-x86-64 python3 \
 && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneapi.list \
 && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-runtime-opencl \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.85.0 \
      --target x86_64-pc-windows-gnu
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends binutils gdb && rm -rf /var/lib/apt/lists/*
