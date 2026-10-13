//! ABI conformance: check ignuda's hand-written driver constants and type
//! widths against ZLUDA's `cuda_types`, which is generated from NVIDIA's public
//! headers. ignuda keeps its own lightweight `CUresult = u32` representation
//! (not `cuda_types`' idiomatic `Result<(), CUerror>`), so this does not swap
//! ignuda's types — it guards their numeric values from drifting.
//!
//! `cuda_types` is a dev-dependency only; the shipped libcuda does not link it.

use cuda::types as ig;
use cuda_types::cuda as ct;

/// ignuda's error codes must equal the authoritative NVIDIA values.
#[test]
fn error_codes_match() {
    // CUDA_SUCCESS is 0 in ignuda; in cuda_types it is `CUresult::Ok(())`.
    assert_eq!(ig::CUDA_SUCCESS, 0);

    macro_rules! chk {
        ($ig:ident, $ct:ident) => {
            assert_eq!(
                ig::$ig,
                ct::CUerror::$ct.0.get(),
                concat!("value mismatch for ", stringify!($ig))
            );
        };
    }
    chk!(CUDA_ERROR_INVALID_VALUE, INVALID_VALUE);
    chk!(CUDA_ERROR_OUT_OF_MEMORY, OUT_OF_MEMORY);
    chk!(CUDA_ERROR_NOT_INITIALIZED, NOT_INITIALIZED);
    chk!(CUDA_ERROR_DEINITIALIZED, DEINITIALIZED);
    chk!(CUDA_ERROR_NO_DEVICE, NO_DEVICE);
    chk!(CUDA_ERROR_INVALID_DEVICE, INVALID_DEVICE);
    chk!(CUDA_ERROR_INVALID_IMAGE, INVALID_IMAGE);
    chk!(CUDA_ERROR_INVALID_CONTEXT, INVALID_CONTEXT);
    chk!(CUDA_ERROR_NO_BINARY_FOR_GPU, NO_BINARY_FOR_GPU);
    chk!(CUDA_ERROR_INVALID_PTX, INVALID_PTX);
    chk!(CUDA_ERROR_INVALID_HANDLE, INVALID_HANDLE);
    chk!(CUDA_ERROR_NOT_FOUND, NOT_FOUND);
    chk!(CUDA_ERROR_NOT_READY, NOT_READY);
    chk!(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, LAUNCH_OUT_OF_RESOURCES);
    chk!(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, PEER_ACCESS_UNSUPPORTED);
    chk!(CUDA_ERROR_NOT_SUPPORTED, NOT_SUPPORTED);
    chk!(CUDA_ERROR_UNKNOWN, UNKNOWN);
}

/// ignuda's driver device-attribute numbers must equal NVIDIA's.
#[test]
fn device_attributes_match() {
    macro_rules! chk {
        ($ig:ident, $ct:ident) => {
            assert_eq!(
                ig::attr::$ig,
                ct::CUdevice_attribute::$ct.0 as i32,
                concat!("value mismatch for attr::", stringify!($ig))
            );
        };
    }
    chk!(MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    chk!(MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    chk!(WARP_SIZE, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    chk!(MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    chk!(MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    chk!(INTEGRATED, CU_DEVICE_ATTRIBUTE_INTEGRATED);
    chk!(UNIFIED_ADDRESSING, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    chk!(COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    chk!(COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    // The higher-numbered attributes are the easiest to mistype by hand:
    chk!(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    chk!(MAX_REGISTERS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    chk!(MANAGED_MEMORY, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    chk!(PAGEABLE_MEMORY_ACCESS, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
    chk!(CONCURRENT_MANAGED_ACCESS, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS);
    chk!(COMPUTE_PREEMPTION_SUPPORTED, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
    chk!(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
}

/// The FFI widths ignuda's exported functions use must match the real ABI.
#[test]
fn abi_widths_match() {
    use std::mem::size_of;
    // CUdeviceptr: ignuda `u64` vs cuda_types newtype over a pointer.
    assert_eq!(size_of::<ig::CUdeviceptr>(), size_of::<ct::CUdeviceptr>());
    assert_eq!(size_of::<ig::CUdeviceptr>(), 8);
    // CUresult crosses the FFI boundary as a 32-bit value in both worlds
    // (cuda_types relies on the NonZeroU32 niche in `Result<(), CUerror>`).
    assert_eq!(size_of::<ig::CUresult>(), 4);
    assert_eq!(size_of::<ct::CUresult>(), 4);
    // CUdevice is a 32-bit ordinal.
    assert_eq!(size_of::<ig::CUdevice>(), size_of::<ct::CUdevice>());
}
