//! ignuda: CUDA driver API (libcuda.so / nvcuda.dll) implemented on top of
//! OpenCL + Intel unified shared memory, with a PTX -> SPIR-V JIT.

pub mod api;
pub mod cl;
pub mod dark;
pub mod driver;
pub mod fatbin;
pub mod types;
