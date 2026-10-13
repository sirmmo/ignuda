//! Driver API entry points, resolved at runtime from libcuda.so.1 / nvcuda.dll
//! so the runtime and the application share one driver instance.
#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::OnceLock;

pub type CUresult = u32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

macro_rules! driver {
    ($( $name:ident ( $($t:ty),* ) ;)*) => {
        pub struct Drv {
            $( pub $name: unsafe extern "system" fn($($t),*) -> CUresult, )*
        }
        impl Drv {
            unsafe fn resolve(lib: *mut c_void) -> Result<Drv, String> {
                Ok(Drv {
                    $( $name: {
                        let p = sym(lib, stringify!($name));
                        if p.is_null() {
                            return Err(format!("driver library lacks {}", stringify!($name)));
                        }
                        std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($t),*) -> CUresult>(p)
                    }, )*
                })
            }
        }
    };
}

driver! {
    cuInit(c_uint);
    cuDriverGetVersion(*mut c_int);
    cuDeviceGetCount(*mut c_int);
    cuDeviceGet(*mut c_int, c_int);
    cuDeviceGetName(*mut c_char, c_int, c_int);
    cuDeviceGetAttribute(*mut c_int, c_int, c_int);
    cuDeviceTotalMem_v2(*mut usize, c_int);
    cuDeviceGetUuid_v2(*mut [u8; 16], c_int);
    cuDeviceGetPCIBusId(*mut c_char, c_int, c_int);
    cuDeviceCanAccessPeer(*mut c_int, c_int, c_int);
    cuDevicePrimaryCtxRetain(*mut CUcontext, c_int);
    cuDevicePrimaryCtxSetFlags_v2(c_int, c_uint);
    cuDevicePrimaryCtxGetState(c_int, *mut c_uint, *mut c_int);
    cuCtxSetCurrent(CUcontext);
    cuCtxGetCurrent(*mut CUcontext);
    cuCtxSynchronize();
    cuCtxGetLimit(*mut usize, c_int);
    cuCtxSetLimit(c_int, usize);
    cuModuleLoadFatBinary(*mut CUmodule, *const c_void);
    cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char);
    cuModuleGetGlobal_v2(*mut u64, *mut usize, CUmodule, *const c_char);
    cuFuncGetAttribute(*mut c_int, c_int, CUfunction);
    cuFuncSetAttribute(CUfunction, c_int, c_int);
    cuLaunchKernel(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(*mut c_int, CUfunction, c_int, usize);
    cuMemAlloc_v2(*mut u64, usize);
    cuMemAllocManaged(*mut u64, usize, c_uint);
    cuMemFree_v2(u64);
    cuMemHostAlloc(*mut *mut c_void, usize, c_uint);
    cuMemFreeHost(*mut c_void);
    cuMemHostRegister_v2(*mut c_void, usize, c_uint);
    cuMemHostUnregister(*mut c_void);
    cuMemHostGetDevicePointer_v2(*mut u64, *mut c_void, c_uint);
    cuMemGetInfo_v2(*mut usize, *mut usize);
    cuMemGetAddressRange_v2(*mut u64, *mut usize, u64);
    cuMemcpy(u64, u64, usize);
    cuMemcpyAsync(u64, u64, usize, CUstream);
    cuMemcpy2D_v2(*const c_void);
    cuMemcpy2DAsync_v2(*const c_void, CUstream);
    cuMemsetD8_v2(u64, u8, usize);
    cuMemsetD8Async(u64, u8, usize, CUstream);
    cuPointerGetAttribute(*mut c_void, c_int, u64);
    cuStreamCreate(*mut CUstream, c_uint);
    cuStreamCreateWithPriority(*mut CUstream, c_uint, c_int);
    cuStreamDestroy_v2(CUstream);
    cuStreamSynchronize(CUstream);
    cuStreamQuery(CUstream);
    cuStreamWaitEvent(CUstream, CUevent, c_uint);
    cuStreamGetPriority(CUstream, *mut c_int);
    cuStreamGetFlags(CUstream, *mut c_uint);
    cuLaunchHostFunc(CUstream, Option<unsafe extern "C" fn(*mut c_void)>, *mut c_void);
    cuEventCreate(*mut CUevent, c_uint);
    cuEventDestroy_v2(CUevent);
    cuEventRecord(CUevent, CUstream);
    cuEventSynchronize(CUevent);
    cuEventQuery(CUevent);
    cuEventElapsedTime(*mut f32, CUevent, CUevent);
    cuGetProcAddress_v2(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int);
}

#[cfg(unix)]
unsafe fn open(name: &str) -> *mut c_void {
    #[link(name = "dl")]
    extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    }
    let n = CString::new(name).unwrap();
    dlopen(n.as_ptr(), 2)
}

#[cfg(unix)]
unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
    #[link(name = "dl")]
    extern "C" {
        fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    let n = CString::new(name).unwrap();
    dlsym(lib, n.as_ptr())
}

#[cfg(windows)]
unsafe fn open(name: &str) -> *mut c_void {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    }
    let n = CString::new(name).unwrap();
    LoadLibraryA(n.as_ptr())
}

#[cfg(windows)]
unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetProcAddress(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    let n = CString::new(name).unwrap();
    GetProcAddress(lib, n.as_ptr())
}

pub fn get() -> Option<&'static Drv> {
    static D: OnceLock<Option<Drv>> = OnceLock::new();
    D.get_or_init(|| unsafe {
        let names: &[&str] = if cfg!(windows) { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };
        for n in names {
            let lib = open(n);
            if !lib.is_null() {
                return match Drv::resolve(lib) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        eprintln!("[ignuda-cudart] {e}");
                        None
                    }
                };
            }
        }
        eprintln!("[ignuda-cudart] could not load the driver library");
        None
    })
    .as_ref()
}
