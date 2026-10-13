//! OpenCL + cl_intel_unified_shared_memory entry points, loaded at runtime
//! (libOpenCL.so.1 / OpenCL.dll) so the library has no link-time dependency.
#![allow(non_camel_case_types, non_snake_case, clippy::type_complexity)]

use std::ffi::{c_char, c_void, CString};
use std::ptr::null_mut;

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_ulong = u64;
pub type cl_bool = u32;
pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_event = *mut c_void;
pub type cl_mem = *mut c_void;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_DEVICE_TYPE_CPU: u64 = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
pub const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;

pub const CL_PLATFORM_NAME: u32 = 0x0902;
pub const CL_DEVICE_TYPE: u32 = 0x1000;
pub const CL_DEVICE_VENDOR_ID: u32 = 0x1001;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
pub const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: u32 = 0x101E;
pub const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
pub const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: u32 = 0x1020;
pub const CL_DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
pub const CL_DEVICE_NAME: u32 = 0x102B;
pub const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
pub const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
pub const CL_DEVICE_UUID_KHR: u32 = 0x106A;
pub const CL_DEVICE_SUB_GROUP_SIZES_INTEL: u32 = 0x4108;
pub const CL_DEVICE_PCI_BUS_INFO_KHR: u32 = 0x410F;
pub const CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT: u32 = 0x4231;
pub const CL_CONTEXT_PLATFORM: isize = 0x1084;
pub const CL_QUEUE_PROPERTIES: u64 = 0x1093;
pub const CL_QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
pub const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
pub const CL_KERNEL_WORK_GROUP_SIZE: u32 = 0x11B0;
pub const CL_KERNEL_LOCAL_MEM_SIZE: u32 = 0x11B2;
pub const CL_KERNEL_PRIVATE_MEM_SIZE: u32 = 0x11B4;
pub const CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL: u32 = 0x4200;
pub const CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL: u32 = 0x4201;
pub const CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL: u32 = 0x4202;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS: u32 = 0x11D3;
pub const CL_PROFILING_COMMAND_START: u32 = 0x1282;
pub const CL_PROFILING_COMMAND_END: u32 = 0x1283;
pub const CL_COMPLETE: cl_int = 0;

macro_rules! cl_api {
    ($( $name:ident : fn($($a:ty),*) -> $r:ty ;)*) => {
        pub struct Cl {
            $( pub $name: unsafe extern "system" fn($($a),*) -> $r, )*
            // cl_intel_unified_shared_memory (resolved per platform)
            pub ext: Vec<(cl_platform_id, Usm)>,
        }
        impl Cl {
            unsafe fn resolve(lib: *mut c_void) -> Result<Cl, String> {
                Ok(Cl {
                    $( $name: {
                        let p = sym(lib, stringify!($name));
                        if p.is_null() { return Err(format!("OpenCL library lacks {}", stringify!($name))); }
                        std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($a),*) -> $r>(p)
                    }, )*
                    ext: Vec::new(),
                })
            }
        }
    };
}

cl_api! {
    clGetPlatformIDs: fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int;
    clGetPlatformInfo: fn(cl_platform_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clGetDeviceIDs: fn(cl_platform_id, u64, cl_uint, *mut cl_device_id, *mut cl_uint) -> cl_int;
    clGetDeviceInfo: fn(cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clCreateContext: fn(*const isize, cl_uint, *const cl_device_id, *const c_void, *mut c_void, *mut cl_int) -> cl_context;
    clCreateCommandQueueWithProperties: fn(cl_context, cl_device_id, *const u64, *mut cl_int) -> cl_command_queue;
    clReleaseCommandQueue: fn(cl_command_queue) -> cl_int;
    clCreateProgramWithIL: fn(cl_context, *const c_void, usize, *mut cl_int) -> cl_program;
    clBuildProgram: fn(cl_program, cl_uint, *const cl_device_id, *const c_char, *const c_void, *mut c_void) -> cl_int;
    clGetProgramBuildInfo: fn(cl_program, cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clReleaseProgram: fn(cl_program) -> cl_int;
    clCreateKernel: fn(cl_program, *const c_char, *mut cl_int) -> cl_kernel;
    clReleaseKernel: fn(cl_kernel) -> cl_int;
    clSetKernelArg: fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int;
    clSetKernelExecInfo: fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int;
    clGetKernelWorkGroupInfo: fn(cl_kernel, cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clEnqueueNDRangeKernel: fn(cl_command_queue, cl_kernel, cl_uint, *const usize, *const usize, *const usize, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clEnqueueMarkerWithWaitList: fn(cl_command_queue, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clEnqueueBarrierWithWaitList: fn(cl_command_queue, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clFinish: fn(cl_command_queue) -> cl_int;
    clFlush: fn(cl_command_queue) -> cl_int;
    clWaitForEvents: fn(cl_uint, *const cl_event) -> cl_int;
    clGetEventInfo: fn(cl_event, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clGetEventProfilingInfo: fn(cl_event, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clReleaseEvent: fn(cl_event) -> cl_int;
    clGetExtensionFunctionAddressForPlatform: fn(cl_platform_id, *const c_char) -> *mut c_void;
}

#[derive(Clone, Copy)]
pub struct Usm {
    pub clHostMemAllocINTEL: unsafe extern "system" fn(cl_context, *const u64, usize, cl_uint, *mut cl_int) -> *mut c_void,
    pub clDeviceMemAllocINTEL:
        unsafe extern "system" fn(cl_context, cl_device_id, *const u64, usize, cl_uint, *mut cl_int) -> *mut c_void,
    pub clSharedMemAllocINTEL:
        unsafe extern "system" fn(cl_context, cl_device_id, *const u64, usize, cl_uint, *mut cl_int) -> *mut c_void,
    pub clMemBlockingFreeINTEL: unsafe extern "system" fn(cl_context, *mut c_void) -> cl_int,
    pub clEnqueueMemcpyINTEL: unsafe extern "system" fn(
        cl_command_queue,
        cl_bool,
        *mut c_void,
        *const c_void,
        usize,
        cl_uint,
        *const cl_event,
        *mut cl_event,
    ) -> cl_int,
    pub clEnqueueMemFillINTEL: unsafe extern "system" fn(
        cl_command_queue,
        *mut c_void,
        *const c_void,
        usize,
        usize,
        cl_uint,
        *const cl_event,
        *mut cl_event,
    ) -> cl_int,
    pub clSetKernelArgMemPointerINTEL: unsafe extern "system" fn(cl_kernel, cl_uint, *const c_void) -> cl_int,
}

#[cfg(unix)]
mod dl {
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "dl")]
    extern "C" {
        pub fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
}

#[cfg(windows)]
mod dl {
    use std::ffi::{c_char, c_void};
    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
}

unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
    let n = CString::new(name).unwrap();
    #[cfg(unix)]
    {
        dl::dlsym(lib, n.as_ptr())
    }
    #[cfg(windows)]
    {
        dl::GetProcAddress(lib, n.as_ptr())
    }
}

unsafe fn open(name: &str) -> *mut c_void {
    let n = CString::new(name).unwrap();
    #[cfg(unix)]
    {
        dl::dlopen(n.as_ptr(), 2 /* RTLD_NOW */)
    }
    #[cfg(windows)]
    {
        dl::LoadLibraryA(n.as_ptr())
    }
}

impl Cl {
    pub fn load() -> Result<Cl, String> {
        let names: &[&str] = if cfg!(windows) { &["OpenCL.dll"] } else { &["libOpenCL.so.1", "libOpenCL.so"] };
        let custom = std::env::var("IGNUDA_OPENCL").ok();
        for n in custom.iter().map(|s| s.as_str()).chain(names.iter().copied()) {
            let lib = unsafe { open(n) };
            if !lib.is_null() {
                return unsafe { Cl::resolve(lib) };
            }
        }
        Err("could not load the OpenCL ICD loader (libOpenCL.so.1 / OpenCL.dll)".into())
    }

    /// Resolve the USM extension functions for a platform.
    pub fn usm(&self, p: cl_platform_id) -> Option<Usm> {
        unsafe {
            let get = |n: &str| {
                let c = CString::new(n).unwrap();
                (self.clGetExtensionFunctionAddressForPlatform)(p, c.as_ptr())
            };
            macro_rules! f {
                ($n:ident) => {{
                    let x = get(stringify!($n));
                    if x.is_null() {
                        return None;
                    }
                    std::mem::transmute(x)
                }};
            }
            Some(Usm {
                clHostMemAllocINTEL: f!(clHostMemAllocINTEL),
                clDeviceMemAllocINTEL: f!(clDeviceMemAllocINTEL),
                clSharedMemAllocINTEL: f!(clSharedMemAllocINTEL),
                clMemBlockingFreeINTEL: f!(clMemBlockingFreeINTEL),
                clEnqueueMemcpyINTEL: f!(clEnqueueMemcpyINTEL),
                clEnqueueMemFillINTEL: f!(clEnqueueMemFillINTEL),
                clSetKernelArgMemPointerINTEL: f!(clSetKernelArgMemPointerINTEL),
            })
        }
    }

    pub fn device_string(&self, d: cl_device_id, param: u32) -> String {
        unsafe {
            let mut n = 0usize;
            (self.clGetDeviceInfo)(d, param, 0, null_mut(), &mut n);
            let mut buf = vec![0u8; n];
            (self.clGetDeviceInfo)(d, param, n, buf.as_mut_ptr() as *mut c_void, null_mut());
            String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
        }
    }

    pub fn device_info<T: Copy + Default>(&self, d: cl_device_id, param: u32) -> T {
        let mut v = T::default();
        unsafe {
            (self.clGetDeviceInfo)(d, param, std::mem::size_of::<T>(), &mut v as *mut T as *mut c_void, null_mut());
        }
        v
    }

    pub fn device_bytes(&self, d: cl_device_id, param: u32) -> Vec<u8> {
        unsafe {
            let mut n = 0usize;
            if (self.clGetDeviceInfo)(d, param, 0, null_mut(), &mut n) != 0 {
                return vec![];
            }
            let mut buf = vec![0u8; n];
            (self.clGetDeviceInfo)(d, param, n, buf.as_mut_ptr() as *mut c_void, null_mut());
            buf
        }
    }
}

unsafe impl Send for Cl {}
unsafe impl Sync for Cl {}
