//! ignuda's CUDA runtime library (libcudart.so.12 / cudart64_12.dll).
//!
//! The runtime is the layer the host side of an nvcc-compiled program talks to.
//! nvcc emits, into every translation unit:
//!   * a fat binary (the embedded device code), registered at startup with
//!     `__cudaRegisterFatBinary`;
//!   * one `__cudaRegisterFunction` call per `__global__`, tying the host-side
//!     stub function's address to a device kernel name;
//!   * `__device__`/`__constant__` variables registered with `__cudaRegisterVar`.
//! A `<<<>>>` launch lowers to `__cudaPushCallConfiguration` then a call to the
//! stub, which calls `cudaLaunchKernel` with its own address.
//!
//! We implement all of this on the ignuda driver API (loaded from libcuda via
//! `drv`), so no NVIDIA component is involved and the integrity-checked
//! NVIDIA runtime is never loaded.

mod drv;
mod rt;

pub use rt::*;

/// Magic of the fat binary wrapper struct nvcc places in the host binary.
pub(crate) fn drv_wrapper_magic() -> u32 {
    0x4662_43B1
}

/// Translate a runtime `cudaDeviceAttr` into the driver `CUdevice_attribute`.
/// The two enums overlap for most values but diverge in a few places; we map
/// the ones PyTorch and typical code query and return -1 for "report zero".
pub(crate) fn map_device_attr(attr: i32) -> i32 {
    // Runtime cudaDevAttr* values; driver CU_DEVICE_ATTRIBUTE_* values.
    match attr {
        1 => 1,   // MaxThreadsPerBlock
        2 => 2,   // MaxBlockDimX
        3 => 3,   // MaxBlockDimY
        4 => 4,   // MaxBlockDimZ
        5 => 5,   // MaxGridDimX
        6 => 6,   // MaxGridDimY
        7 => 7,   // MaxGridDimZ
        8 => 8,   // MaxSharedMemoryPerBlock
        9 => 9,   // TotalConstantMemory
        10 => 10, // WarpSize
        11 => 11, // MaxPitch
        12 => 12, // MaxRegistersPerBlock
        13 => 13, // ClockRate
        14 => 14, // TextureAlignment
        15 => 15, // GpuOverlap
        16 => 16, // MultiProcessorCount
        17 => 17, // KernelExecTimeout
        18 => 18, // Integrated
        19 => 19, // CanMapHostMemory
        20 => 20, // ComputeMode
        36 => 36, // MemoryClockRate
        37 => 37, // GlobalMemoryBusWidth
        38 => 38, // L2CacheSize
        39 => 39, // MaxThreadsPerMultiProcessor
        40 => 40, // AsyncEngineCount
        41 => 41, // UnifiedAddressing
        75 => 75, // ComputeCapabilityMajor
        76 => 76, // ComputeCapabilityMinor
        82 => 81, // MaxSharedMemoryPerMultiprocessor
        83 => 82, // MaxRegistersPerMultiprocessor
        84 => 83, // ManagedMemory
        86 => 88, // PageableMemoryAccess
        87 => 89, // ConcurrentManagedAccess
        90 => 97, // MaxSharedMemoryPerBlockOptin
        _ => -1,
    }
}
