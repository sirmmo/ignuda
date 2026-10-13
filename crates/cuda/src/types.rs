//! CUDA driver API types and constants (subset).
#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type CUresult = u32;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUlibrary = *mut c_void;
pub type CUkernel = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701;
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: CUresult = 217;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

pub fn error_name(e: CUresult) -> &'static str {
    match e {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        217 => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        218 => "CUDA_ERROR_INVALID_PTX",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_FOUND",
        600 => "CUDA_ERROR_NOT_READY",
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}

pub fn error_string(e: CUresult) -> &'static str {
    match e {
        0 => "no error",
        1 => "invalid argument",
        2 => "out of memory",
        3 => "initialization error",
        100 => "no CUDA-capable device is detected",
        101 => "invalid device ordinal",
        200 => "device kernel image is invalid",
        201 => "invalid device context",
        209 => "no kernel image is available for execution on the device (ignuda needs PTX)",
        218 => "a PTX JIT compilation failed (set IGNUDA_LOG=1 for details)",
        400 => "invalid resource handle",
        500 => "named symbol not found",
        600 => "device not ready",
        701 => "too many resources requested for launch",
        801 => "operation not supported",
        _ => "unknown error",
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: u32,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: *mut c_void,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: u32,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: *mut c_void,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

#[repr(C)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

// CUdevice_attribute values
pub mod attr {
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const MAX_BLOCK_DIM_X: i32 = 2;
    pub const MAX_BLOCK_DIM_Y: i32 = 3;
    pub const MAX_BLOCK_DIM_Z: i32 = 4;
    pub const MAX_GRID_DIM_X: i32 = 5;
    pub const MAX_GRID_DIM_Y: i32 = 6;
    pub const MAX_GRID_DIM_Z: i32 = 7;
    pub const MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
    pub const TOTAL_CONSTANT_MEMORY: i32 = 9;
    pub const WARP_SIZE: i32 = 10;
    pub const MAX_PITCH: i32 = 11;
    pub const MAX_REGISTERS_PER_BLOCK: i32 = 12;
    pub const CLOCK_RATE: i32 = 13;
    pub const TEXTURE_ALIGNMENT: i32 = 14;
    pub const GPU_OVERLAP: i32 = 15;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const KERNEL_EXEC_TIMEOUT: i32 = 17;
    pub const INTEGRATED: i32 = 18;
    pub const CAN_MAP_HOST_MEMORY: i32 = 19;
    pub const COMPUTE_MODE: i32 = 20;
    pub const CONCURRENT_KERNELS: i32 = 31;
    pub const ECC_ENABLED: i32 = 32;
    pub const PCI_BUS_ID: i32 = 33;
    pub const PCI_DEVICE_ID: i32 = 34;
    pub const TCC_DRIVER: i32 = 35;
    pub const MEMORY_CLOCK_RATE: i32 = 36;
    pub const GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
    pub const L2_CACHE_SIZE: i32 = 38;
    pub const MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
    pub const ASYNC_ENGINE_COUNT: i32 = 40;
    pub const UNIFIED_ADDRESSING: i32 = 41;
    pub const PCI_DOMAIN_ID: i32 = 50;
    pub const TEXTURE_PITCH_ALIGNMENT: i32 = 51;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const STREAM_PRIORITIES_SUPPORTED: i32 = 78;
    pub const GLOBAL_L1_CACHE_SUPPORTED: i32 = 79;
    pub const LOCAL_L1_CACHE_SUPPORTED: i32 = 80;
    pub const MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;
    pub const MAX_REGISTERS_PER_MULTIPROCESSOR: i32 = 82;
    pub const MANAGED_MEMORY: i32 = 83;
    pub const MULTI_GPU_BOARD: i32 = 84;
    pub const SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: i32 = 87;
    pub const PAGEABLE_MEMORY_ACCESS: i32 = 88;
    pub const CONCURRENT_MANAGED_ACCESS: i32 = 89;
    pub const COMPUTE_PREEMPTION_SUPPORTED: i32 = 90;
    pub const CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: i32 = 91;
    pub const COOPERATIVE_LAUNCH: i32 = 95;
    pub const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;
    pub const CAN_FLUSH_REMOTE_WRITES: i32 = 98;
    pub const HOST_REGISTER_SUPPORTED: i32 = 99;
    pub const PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: i32 = 100;
    pub const DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: i32 = 101;
    pub const VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: i32 = 102;
    pub const MAX_BLOCKS_PER_MULTIPROCESSOR: i32 = 106;
    pub const RESERVED_SHARED_MEMORY_PER_BLOCK: i32 = 111;
    pub const MEMORY_POOLS_SUPPORTED: i32 = 115;
}
