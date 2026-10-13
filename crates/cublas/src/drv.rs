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
    cuModuleLoadData(*mut CUmodule, *const c_void);
    cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char);
    cuLaunchKernel(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void);
    cuMemcpy(u64, u64, usize);
    cuCtxSynchronize();
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
