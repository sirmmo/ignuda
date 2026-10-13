//! cuGetExportTable: NVIDIA's undocumented "dark API" tables used by cudart.
//!
//! Table layouts follow what the CUDA runtime is known to call (as also
//! reverse-engineered by ZLUDA). Unknown slots point to logging stubs so new
//! cudart versions show what they need instead of crashing silently.
#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use crate::driver::{self, log, Res};
use crate::types::*;

type Guid = [u8; 16];

const fn guid(s: &str) -> Guid {
    // "6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9" (byte order as found in memory)
    let b = s.as_bytes();
    let mut out = [0u8; 16];
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        if b[i] == b'-' {
            i += 1;
            continue;
        }
        let hi = hex(b[i]);
        let lo = hex(b[i + 1]);
        out[n] = hi * 16 + lo;
        n += 1;
        i += 2;
    }
    out
}

const fn hex(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => c - b'A' + 10,
    }
}

const CUDART_INTERFACE: Guid = guid("6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9");
const TOOLS_TLS: Guid = guid("42D85A81-23F6-CB47-8298-F6E78A3AECDC");
const TOOLS_RUNTIME_CALLBACK_HOOKS: Guid = guid("A094798C-2E74-2E74-93F2-0800200C0A66");
const CONTEXT_LOCAL_STORAGE: Guid = guid("C693336E-1121-DF11-A8C3-68F355D89593");
const CTX_CREATE_BYPASS: Guid = guid("0CA50B8C-1004-929A-89A7-D0DF10E77286");
const HEAP_ACCESS: Guid = guid("195BCBF4-D67D-0249-ACDE-1DCF5E6DEA6B");
const DEVICE_EXTENDED_RT: Guid = guid("B10541E1-F7C7-C74A-9F64-F223BE99F1E2");
const INTEGRITY_CHECK: Guid = guid("D4082055-BDE6-704B-8D34-BA123C66E1F2");
const UNKNOWN_CHECKS: Guid = guid("263E8860-7CD2-6143-92F6-BBD5006DFA7E");

// ---------------------------------------------------------------- stubs

macro_rules! stubs {
    ($($name:ident = $label:expr;)*) => {
        $(
            unsafe extern "system" fn $name(a: usize, b: usize, c: usize, d: usize) -> u32 {
                log!("dark api {} called ({a:#x}, {b:#x}, {c:#x}, {d:#x}) — unimplemented, returning success", $label);
                0
            }
        )*
    };
}

stubs! {
    s_rt3 = "CUDART_INTERFACE[3]"; s_rt4 = "CUDART_INTERFACE[4]"; s_rt5 = "CUDART_INTERFACE[5]";
    s_rt9 = "CUDART_INTERFACE[9]";
    s_tls1 = "TOOLS_TLS[1]"; s_tls2 = "TOOLS_TLS[2]"; s_tls3 = "TOOLS_TLS[3]";
    s_hook1 = "CALLBACK_HOOKS[1]"; s_hook3 = "CALLBACK_HOOKS[3]"; s_hook4 = "CALLBACK_HOOKS[4]"; s_hook5 = "CALLBACK_HOOKS[5]";
    s_cls3 = "CONTEXT_LOCAL_STORAGE[3]";
    s_ext = "DEVICE_EXTENDED_RT[?]";
    s_unk0 = "UNKNOWN_CHECKS[0]"; s_unk1 = "UNKNOWN_CHECKS[1]";
}

// ------------------------------------------------------ CUDART_INTERFACE

unsafe extern "system" fn get_module_from_cubin(module: *mut CUmodule, wrapper: *const c_void) -> CUresult {
    load(module, wrapper, "get_module_from_cubin")
}

unsafe extern "system" fn get_module_from_cubin_ext1(
    module: *mut CUmodule,
    wrapper: *const c_void,
    _a: *mut c_void,
    _b: *mut c_void,
    _c: c_uint,
) -> CUresult {
    load(module, wrapper, "get_module_from_cubin_ext1")
}

unsafe extern "system" fn get_module_from_cubin_ext2(
    fatbin: *const c_void,
    module: *mut CUmodule,
    _a: *mut c_void,
    _b: *mut c_void,
    _c: c_uint,
) -> CUresult {
    load(module, fatbin, "get_module_from_cubin_ext2")
}

unsafe fn load(module: *mut CUmodule, image: *const c_void, what: &str) -> CUresult {
    let r = (|| -> Res {
        let ctx = driver::current()?;
        let m = driver::load_image(ctx.dev, image)?;
        *module = Box::into_raw(m) as CUmodule;
        Ok(())
    })();
    log!("dark {what} -> {r:?}");
    r.err().unwrap_or(0)
}

unsafe extern "system" fn cudart_fn2(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    match driver::primary_retain(dev) {
        Ok(c) => {
            *pctx = c as CUcontext;
            0
        }
        Err(e) => e,
    }
}

unsafe extern "system" fn cudart_fn7(_a: usize) -> CUresult {
    0
}

// ------------------------------------------------- context local storage

type ClsKey = (usize, usize);
fn cls() -> &'static Mutex<HashMap<ClsKey, (usize, usize)>> {
    static M: OnceLock<Mutex<HashMap<ClsKey, (usize, usize)>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ctx_or_current(c: CUcontext) -> usize {
    if c.is_null() {
        driver::current().map_or(0, |c| c as *const _ as usize)
    } else {
        c as usize
    }
}

unsafe extern "system" fn cls_ctor(ctx: CUcontext, manager: *mut c_void, state: *mut c_void, dtor: usize) -> CUresult {
    cls().lock().unwrap().insert((ctx_or_current(ctx), manager as usize), (state as usize, dtor));
    0
}

unsafe extern "system" fn cls_dtor(ctx: *mut c_void, manager: *mut c_void) -> CUresult {
    cls().lock().unwrap().remove(&(ctx_or_current(ctx), manager as usize));
    0
}

unsafe extern "system" fn cls_get_state(out: *mut *mut c_void, ctx: CUcontext, manager: *mut c_void) -> CUresult {
    match cls().lock().unwrap().get(&(ctx_or_current(ctx), manager as usize)) {
        Some(&(s, _)) => {
            *out = s as *mut c_void;
            0
        }
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}

// --------------------------------------------------------------- others

unsafe extern "system" fn ctx_create_bypass(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult {
    crate::api::cuCtxCreate_v2(pctx, flags, dev)
}

unsafe extern "system" fn heap_alloc(_rec: *mut *const c_void, _a: usize, _b: usize) -> CUresult {
    log!("dark heap_alloc unimplemented");
    CUDA_ERROR_NOT_SUPPORTED
}

unsafe extern "system" fn heap_free(_rec: *const c_void, _a: *mut usize) -> CUresult {
    CUDA_ERROR_NOT_SUPPORTED
}

unsafe extern "system" fn device_get_attribute_ext(_dev: CUdevice, attr: c_uint, _u: c_int, out: *mut [usize; 2]) -> CUresult {
    log!("dark device_get_attribute_ext({attr})");
    if !out.is_null() {
        *out = [0, 0];
    }
    0
}

unsafe extern "system" fn device_get_something(out: *mut u8, _dev: CUdevice) -> CUresult {
    if !out.is_null() {
        *out = 0;
    }
    0
}

unsafe extern "system" fn integrity_check(version: u32, secs: u64, out: *mut [u64; 2]) -> CUresult {
    log!("dark integrity_check(version {version}, time {secs}) — returning zeros");
    if !out.is_null() {
        *out = [0, 0];
    }
    0
}

unsafe extern "system" fn context_check(_ctx: CUcontext, r1: *mut u32, r2: *mut *const c_void) -> CUresult {
    if !r1.is_null() {
        *r1 = 0;
    }
    if !r2.is_null() {
        *r2 = std::ptr::null();
    }
    0
}

unsafe extern "system" fn check_fn3() -> u32 {
    0
}

unsafe extern "system" fn hooks_fn2(ptr: *mut *mut usize, size: *mut usize) {
    static mut HOOKS2: [usize; 512] = [0; 512];
    *ptr = std::ptr::addr_of_mut!(HOOKS2) as *mut usize;
    *size = 512 * 8;
}

unsafe extern "system" fn hooks_fn6(ptr: *mut *mut usize, size: *mut usize) {
    static mut HOOKS6: [usize; 2] = [0; 2];
    *ptr = std::ptr::addr_of_mut!(HOOKS6) as *mut usize;
    *size = 2 * 8;
}

// ---------------------------------------------------------------- tables

fn tables() -> &'static HashMap<Guid, Vec<usize>> {
    static T: OnceLock<HashMap<Guid, Vec<usize>>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = HashMap::new();
        let sized = |v: Vec<usize>| {
            let mut v = v;
            v[0] = v.len() * std::mem::size_of::<usize>();
            v
        };
        m.insert(
            CUDART_INTERFACE,
            sized(vec![
                0,
                get_module_from_cubin as usize,
                cudart_fn2 as usize,
                s_rt3 as usize,
                s_rt4 as usize,
                s_rt5 as usize,
                get_module_from_cubin_ext1 as usize,
                cudart_fn7 as usize,
                get_module_from_cubin_ext2 as usize,
                s_rt9 as usize,
            ]),
        );
        m.insert(TOOLS_TLS, sized(vec![0, s_tls1 as usize, s_tls2 as usize, s_tls3 as usize]));
        m.insert(
            TOOLS_RUNTIME_CALLBACK_HOOKS,
            sized(vec![
                0,
                s_hook1 as usize,
                hooks_fn2 as usize,
                s_hook3 as usize,
                s_hook4 as usize,
                s_hook5 as usize,
                hooks_fn6 as usize,
            ]),
        );
        m.insert(CONTEXT_LOCAL_STORAGE, vec![cls_ctor as usize, cls_dtor as usize, cls_get_state as usize, s_cls3 as usize]);
        m.insert(CTX_CREATE_BYPASS, sized(vec![0, ctx_create_bypass as usize]));
        m.insert(HEAP_ACCESS, sized(vec![0, heap_alloc as usize, heap_free as usize]));
        let mut ext = vec![s_ext as usize; 26];
        ext[5] = device_get_attribute_ext as usize;
        ext[13] = device_get_something as usize;
        m.insert(DEVICE_EXTENDED_RT, sized(ext));
        m.insert(INTEGRITY_CHECK, sized(vec![0, integrity_check as usize, s_ext as usize]));
        m.insert(UNKNOWN_CHECKS, vec![s_unk0 as usize, s_unk1 as usize, context_check as usize, check_fn3 as usize]);
        m
    })
}

pub unsafe fn export_table(table: *mut *const c_void, id: *const CUuuid) -> Res {
    if table.is_null() || id.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let u = (*id).bytes;
    match tables().get(&u) {
        Some(t) if !(u == INTEGRITY_CHECK && std::env::var_os("IGNUDA_NO_INTEGRITY").is_some()) => {
            log!("cuGetExportTable {{{}}}", u.iter().map(|b| format!("{b:02x}")).collect::<String>());
            *table = t.as_ptr() as *const c_void;
            Ok(())
        }
        _ => {
            log!("cuGetExportTable {{{}}} unknown", u.iter().map(|b| format!("{b:02x}")).collect::<String>());
            *table = std::ptr::null();
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }
}
