//! ignuda's cuBLAS shim (libcublas.so.12).
//!
//! A minimal, correctness-first BLAS on top of the ignuda driver API: the
//! kernels are plain PTX (compiled from `kernels/gemm.cu`) JIT'd through
//! libcuda, so no NVIDIA component and no external math library is involved.
//! Performance work (tiling, or dispatching to Intel oneMKL) comes later;
//! the goal here is to let a cuBLAS-using program — PyTorch's linear layers
//! included — run and produce correct results.
#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc, clippy::too_many_arguments)]

mod drv;

use std::ffi::{c_int, c_void};
use std::ptr::null_mut;
use std::sync::Mutex;

use drv::{CUfunction, CUmodule, CUstream, Drv};

pub type cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;

pub type cublasHandle_t = *mut c_void;
pub type cublasOperation_t = c_int; // 0 = N, 1 = T, 2 = C

const GEMM_PTX: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels/gemm.ptx"));

struct Kernels {
    module: CUmodule,
    sgemm: CUfunction,
    dgemm: CUfunction,
    saxpy: CUfunction,
    sscal: CUfunction,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn kernels() -> Result<&'static Kernels, cublasStatus_t> {
    static K: Mutex<Option<&'static Kernels>> = Mutex::new(None);
    let mut g = K.lock().unwrap();
    if let Some(k) = *g {
        return Ok(k);
    }
    let drv = drv::get().ok_or(CUBLAS_STATUS_NOT_INITIALIZED)?;
    unsafe {
        // The caller's context/primary context is already current (cublasCreate
        // is always called after the runtime has bound a device).
        let mut module: CUmodule = null_mut();
        let ptx = std::ffi::CString::new(GEMM_PTX).unwrap();
        if (drv.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void) != 0 {
            return Err(CUBLAS_STATUS_INTERNAL_ERROR);
        }
        let get = |name: &str| -> CUfunction {
            let n = std::ffi::CString::new(name).unwrap();
            let mut f: CUfunction = null_mut();
            (drv.cuModuleGetFunction)(&mut f, module, n.as_ptr());
            f
        };
        let k = Box::leak(Box::new(Kernels {
            module,
            sgemm: get("sgemm"),
            dgemm: get("dgemm"),
            saxpy: get("saxpy"),
            sscal: get("sscal"),
        }));
        if k.sgemm.is_null() {
            return Err(CUBLAS_STATUS_INTERNAL_ERROR);
        }
        *g = Some(k);
        Ok(k)
    }
}

struct Handle {
    stream: CUstream,
    pointer_mode_device: bool,
}

#[no_mangle]
pub unsafe extern "C" fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if kernels().is_err() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *handle = Box::into_raw(Box::new(Handle { stream: null_mut(), pointer_mode_device: false })) as cublasHandle_t;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut Handle));
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetVersion_v2(_handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t {
    if !version.is_null() {
        *version = 120400;
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t {
    if handle.is_null() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    (*(handle as *mut Handle)).stream = stream;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetStream_v2(handle: cublasHandle_t, stream: *mut CUstream) -> cublasStatus_t {
    if handle.is_null() || stream.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *stream = (*(handle as *mut Handle)).stream;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: c_int) -> cublasStatus_t {
    if handle.is_null() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    (*(handle as *mut Handle)).pointer_mode_device = mode == 1;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut c_int) -> cublasStatus_t {
    if handle.is_null() || mode.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = (*(handle as *mut Handle)).pointer_mode_device as c_int;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetMathMode(_handle: cublasHandle_t, _mode: c_int) -> cublasStatus_t {
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetWorkspace_v2(_h: cublasHandle_t, _ws: *mut c_void, _sz: usize) -> cublasStatus_t {
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetProperty(ty: c_int, value: *mut c_int) -> cublasStatus_t {
    if value.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // MAJOR=0, MINOR=1, PATCH=2
    *value = match ty {
        0 => 12,
        1 => 4,
        _ => 0,
    };
    CUBLAS_STATUS_SUCCESS
}

/// Read a scalar (alpha/beta) that may live in host or device memory.
unsafe fn scalar<T: Copy>(h: &Handle, p: *const T) -> T {
    if h.pointer_mode_device {
        // bring it back from the device
        let mut v = std::mem::MaybeUninit::<T>::uninit();
        if let Some(drv) = drv::get() {
            (drv.cuMemcpy)(v.as_mut_ptr() as u64, p as u64, std::mem::size_of::<T>());
            (drv.cuCtxSynchronize)();
        }
        v.assume_init()
    } else {
        *p
    }
}

unsafe fn launch_gemm(
    handle: cublasHandle_t,
    f: CUfunction,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    b: u64,
    ldb: c_int,
    beta: *const c_void,
    c: u64,
    ldc: c_int,
    double: bool,
) -> cublasStatus_t {
    if handle.is_null() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    let drv = match drv::get() {
        Some(d) => d,
        None => return CUBLAS_STATUS_NOT_INITIALIZED,
    };
    let h = &*(handle as *const Handle);
    let mut ta: c_int = (transa != 0) as c_int;
    let mut tb: c_int = (transb != 0) as c_int;
    let mut m = m;
    let mut n = n;
    let mut k = k;
    let (mut lda, mut ldb, mut ldc) = (lda, ldb, ldc);

    // Build the kernel argument list. GEMM arguments are passed by pointer.
    let block = [16u32, 16, 1];
    let grid = [((m as u32) + 15) / 16, ((n as u32) + 15) / 16, 1];
    let mut a_ptr = a;
    let mut b_ptr = b;
    let c_ptr = c;

    macro_rules! run {
        ($af:ty) => {{
            let alpha_v: $af = scalar(h, alpha as *const $af);
            let beta_v: $af = scalar(h, beta as *const $af);
            let mut args: Vec<*mut c_void> = vec![
                &mut (ta) as *mut _ as *mut c_void,
                &mut (tb) as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut k as *mut _ as *mut c_void,
                &alpha_v as *const _ as *mut c_void,
                &mut a_ptr as *mut _ as *mut c_void,
                &mut lda as *mut _ as *mut c_void,
                &mut b_ptr as *mut _ as *mut c_void,
                &mut ldb as *mut _ as *mut c_void,
                &beta_v as *const _ as *mut c_void,
                &c_ptr as *const _ as *mut c_void,
                &mut ldc as *mut _ as *mut c_void,
            ];
            let r = (drv.cuLaunchKernel)(
                f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, h.stream,
                args.as_mut_ptr(), null_mut(),
            );
            if r != 0 {
                return CUBLAS_STATUS_EXECUTION_FAILED;
            }
        }};
    }
    if double {
        run!(f64);
    } else {
        run!(f32);
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> cublasStatus_t {
    let k_ = match kernels() {
        Ok(k) => k,
        Err(e) => return e,
    };
    launch_gemm(
        handle, k_.sgemm, transa, transb, m, n, k, alpha as *const c_void, a as u64, lda, b as u64, ldb,
        beta as *const c_void, c as u64, ldc, false,
    )
}

#[no_mangle]
pub unsafe extern "C" fn cublasDgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> cublasStatus_t {
    let k_ = match kernels() {
        Ok(k) => k,
        Err(e) => return e,
    };
    launch_gemm(
        handle, k_.dgemm, transa, transb, m, n, k, alpha as *const c_void, a as u64, lda, b as u64, ldb,
        beta as *const c_void, c as u64, ldc, true,
    )
}

#[no_mangle]
pub unsafe extern "C" fn cublasSaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> cublasStatus_t {
    let k = match kernels() {
        Ok(k) => k,
        Err(e) => return e,
    };
    if handle.is_null() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    let drv = drv::get().unwrap();
    let h = &*(handle as *const Handle);
    let mut n = n;
    let a = scalar(h, alpha);
    let mut xp = x as u64;
    let yp = y as u64;
    let mut incx = incx;
    let mut incy = incy;
    let mut args: Vec<*mut c_void> = vec![
        &mut n as *mut _ as *mut c_void,
        &a as *const _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut incx as *mut _ as *mut c_void,
        &yp as *const _ as *mut c_void,
        &mut incy as *mut _ as *mut c_void,
    ];
    let g = ((n as u32) + 255) / 256;
    let r = (drv.cuLaunchKernel)(k.saxpy, g, 1, 1, 256, 1, 1, 0, h.stream, args.as_mut_ptr(), null_mut());
    if r != 0 {
        CUBLAS_STATUS_EXECUTION_FAILED
    } else {
        CUBLAS_STATUS_SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasSscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: c_int,
) -> cublasStatus_t {
    let k = match kernels() {
        Ok(k) => k,
        Err(e) => return e,
    };
    if handle.is_null() {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    let drv = drv::get().unwrap();
    let h = &*(handle as *const Handle);
    let mut n = n;
    let a = scalar(h, alpha);
    let xp = x as u64;
    let mut incx = incx;
    let mut args: Vec<*mut c_void> = vec![
        &mut n as *mut _ as *mut c_void,
        &a as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &mut incx as *mut _ as *mut c_void,
    ];
    let g = ((n as u32) + 255) / 256;
    let r = (drv.cuLaunchKernel)(k.sscal, g, 1, 1, 256, 1, 1, 0, h.stream, args.as_mut_ptr(), null_mut());
    if r != 0 {
        CUBLAS_STATUS_EXECUTION_FAILED
    } else {
        CUBLAS_STATUS_SUCCESS
    }
}
