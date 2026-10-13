//! Exported CUDA driver API entry points.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::ptr::null_mut;
use std::sync::atomic::Ordering;

use crate::cl::*;
use crate::driver::{self, *};
use crate::types::*;

fn run(name: &str, f: impl FnOnce() -> Res) -> CUresult {
    let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => CUDA_SUCCESS,
        Ok(Err(e)) => e,
        Err(_) => {
            eprintln!("[ignuda] panic in {name}");
            CUDA_ERROR_UNKNOWN
        }
    };
    if driver::log_enabled() {
        if r == CUDA_SUCCESS {
            eprintln!("[ignuda] {name}");
        } else {
            eprintln!("[ignuda] {name} -> {} ({r})", error_name(r));
        }
    }
    r
}

macro_rules! out {
    ($p:expr) => {{
        if $p.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        &mut *$p
    }};
}

macro_rules! api {
    ($( fn $name:ident ( $($a:ident : $t:ty),* $(,)? ) $body:block )*) => {
        $(
            #[no_mangle]
            pub unsafe extern "system" fn $name($($a: $t),*) -> CUresult {
                run(stringify!($name), || -> Res { $body })
            }
        )*
        /// (name, address) of every exported entry point, for cuGetProcAddress.
        pub fn table() -> Vec<(&'static str, usize)> {
            vec![ $( (stringify!($name), $name as usize) ),* ]
        }
    };
}

fn ctx_ptr(c: CUcontext) -> Res<&'static Context> {
    if c.is_null() {
        return Err(CUDA_ERROR_INVALID_CONTEXT);
    }
    Ok(unsafe { &*(c as *const Context) })
}

fn func(f: CUfunction) -> Res<&'static Function> {
    if f.is_null() {
        return Err(CUDA_ERROR_INVALID_HANDLE);
    }
    Ok(unsafe { &*(f as *const Function) })
}

fn module(m: CUmodule) -> Res<&'static Module> {
    if m.is_null() {
        return Err(CUDA_ERROR_INVALID_HANDLE);
    }
    Ok(unsafe { &*(m as *const Module) })
}

unsafe fn load_module(out: *mut CUmodule, image: *const c_void) -> Res {
    let ctx = current()?;
    let m = load_image(ctx.dev, image)?;
    *out!(out) = Box::into_raw(m) as CUmodule;
    Ok(())
}

fn device_attribute(dev: &Device, a: i32) -> i32 {
    use crate::types::attr::*;
    let d = drv().unwrap();
    let max_wg = dev.max_wg.min(1024) as i32;
    match a {
        MAX_THREADS_PER_BLOCK => max_wg,
        MAX_BLOCK_DIM_X | MAX_BLOCK_DIM_Y => max_wg,
        MAX_BLOCK_DIM_Z => 64.min(max_wg),
        MAX_GRID_DIM_X => i32::MAX,
        MAX_GRID_DIM_Y | MAX_GRID_DIM_Z => 65535,
        MAX_SHARED_MEMORY_PER_BLOCK | MAX_SHARED_MEMORY_PER_BLOCK_OPTIN | MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => {
            dev.local_mem as i32
        }
        TOTAL_CONSTANT_MEMORY => dev.const_mem.min(i32::MAX as u64) as i32,
        WARP_SIZE => 32,
        MAX_PITCH => i32::MAX,
        MAX_REGISTERS_PER_BLOCK | MAX_REGISTERS_PER_MULTIPROCESSOR => 65536,
        CLOCK_RATE => dev.clock_mhz as i32 * 1000,
        MEMORY_CLOCK_RATE => 2_400_000,
        TEXTURE_ALIGNMENT => 512,
        TEXTURE_PITCH_ALIGNMENT => 32,
        GPU_OVERLAP | ASYNC_ENGINE_COUNT => 1,
        MULTIPROCESSOR_COUNT => dev.compute_units as i32,
        KERNEL_EXEC_TIMEOUT => 0,
        INTEGRATED => dev.is_gpu as i32,
        CAN_MAP_HOST_MEMORY | UNIFIED_ADDRESSING | MANAGED_MEMORY | CONCURRENT_KERNELS | GLOBAL_L1_CACHE_SUPPORTED
        | LOCAL_L1_CACHE_SUPPORTED | HOST_REGISTER_SUPPORTED | CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM => 1,
        COMPUTE_MODE | ECC_ENABLED | TCC_DRIVER | MULTI_GPU_BOARD | STREAM_PRIORITIES_SUPPORTED => 0,
        PCI_DOMAIN_ID => dev.pci[0] as i32,
        PCI_BUS_ID => dev.pci[1] as i32,
        PCI_DEVICE_ID => dev.pci[2] as i32,
        GLOBAL_MEMORY_BUS_WIDTH => 128,
        L2_CACHE_SIZE => dev.l2.min(i32::MAX as u64) as i32,
        MAX_THREADS_PER_MULTIPROCESSOR => (max_wg * 2).max(1024),
        MAX_BLOCKS_PER_MULTIPROCESSOR => 16,
        COMPUTE_CAPABILITY_MAJOR => d.cc.0,
        COMPUTE_CAPABILITY_MINOR => d.cc.1,
        SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO => 32,
        PAGEABLE_MEMORY_ACCESS | CONCURRENT_MANAGED_ACCESS | COMPUTE_PREEMPTION_SUPPORTED | COOPERATIVE_LAUNCH => 0,
        RESERVED_SHARED_MEMORY_PER_BLOCK => 0,
        MEMORY_POOLS_SUPPORTED | VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED => 0,
        _ => 0,
    }
}

fn memcpy2d(p: &CUDA_MEMCPY2D, stream: CUstream, blocking: bool) -> Res {
    // memory types: 1 host, 2 device, 3 array, 4 unified
    if p.srcMemoryType == 3 || p.dstMemoryType == 3 {
        return Err(CUDA_ERROR_NOT_SUPPORTED);
    }
    let q = queue_of(stream)?;
    let src = if p.srcMemoryType == 1 { p.srcHost as u64 } else { p.srcDevice };
    let dst = if p.dstMemoryType == 1 { p.dstHost as u64 } else { p.dstDevice };
    let spitch = if p.srcPitch == 0 { p.WidthInBytes } else { p.srcPitch };
    let dpitch = if p.dstPitch == 0 { p.WidthInBytes } else { p.dstPitch };
    for row in 0..p.Height {
        let s = src + ((p.srcY + row) * spitch + p.srcXInBytes) as u64;
        let d = dst + ((p.dstY + row) * dpitch + p.dstXInBytes) as u64;
        memcpy(d, s, p.WidthInBytes, q, false)?;
    }
    if blocking {
        current()?.finish()?;
    }
    Ok(())
}

fn ptx_version() -> i32 {
    12040
}

api! {
    // ------------------------------------------------------------ init/device
    fn cuInit(_flags: c_uint) {
        driver::init()
    }

    fn cuDriverGetVersion(v: *mut c_int) {
        *out!(v) = ptx_version();
        Ok(())
    }

    fn cuDeviceGetCount(n: *mut c_int) {
        *out!(n) = drv()?.devices.len() as c_int;
        Ok(())
    }

    fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) {
        drv()?.device(ordinal)?;
        *out!(dev) = ordinal;
        Ok(())
    }

    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) {
        let d = drv()?.device(dev)?;
        if name.is_null() || len <= 0 {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let s = format!("{} [ignuda]", d.name);
        let n = s.len().min(len as usize - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr(), name as *mut u8, n);
        *name.add(n) = 0;
        Ok(())
    }

    fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) {
        out!(uuid).bytes = drv()?.device(dev)?.uuid;
        Ok(())
    }

    fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) {
        out!(uuid).bytes = drv()?.device(dev)?.uuid;
        Ok(())
    }

    fn cuDeviceGetLuid(luid: *mut c_char, mask: *mut c_uint, dev: CUdevice) {
        drv()?.device(dev)?;
        std::ptr::write_bytes(out!(luid) as *mut c_char, 0, 8);
        *out!(mask) = 1;
        Ok(())
    }

    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) {
        *out!(bytes) = drv()?.device(dev)?.total_mem as usize;
        Ok(())
    }

    fn cuDeviceGetAttribute(v: *mut c_int, a: c_int, dev: CUdevice) {
        let d = drv()?.device(dev)?;
        *out!(v) = device_attribute(d, a);
        Ok(())
    }

    fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) {
        drv()?.device(dev)?;
        *out!(major) = drv()?.cc.0;
        *out!(minor) = drv()?.cc.1;
        Ok(())
    }

    fn cuDeviceGetPCIBusId(buf: *mut c_char, len: c_int, dev: CUdevice) {
        let d = drv()?.device(dev)?;
        let s = format!("{:04x}:{:02x}:{:02x}.{:x}", d.pci[0], d.pci[1], d.pci[2], d.pci[3]);
        let n = s.len().min((len as usize).saturating_sub(1));
        std::ptr::copy_nonoverlapping(s.as_ptr(), out!(buf) as *mut c_char as *mut u8, n);
        *buf.add(n) = 0;
        Ok(())
    }

    fn cuDeviceCanAccessPeer(can: *mut c_int, _a: CUdevice, _b: CUdevice) {
        *out!(can) = 0;
        Ok(())
    }

    fn cuDeviceGetP2PAttribute(v: *mut c_int, _attr: c_int, _a: CUdevice, _b: CUdevice) {
        *out!(v) = 0;
        Ok(())
    }

    fn cuDeviceGetTexture1DLinearMaxWidth(w: *mut usize, _fmt: c_int, _ch: c_uint, _dev: CUdevice) {
        *out!(w) = 1 << 27;
        Ok(())
    }

    // -------------------------------------------------------- primary context
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) {
        *out!(ctx) = primary_retain(dev)? as CUcontext;
        Ok(())
    }

    fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) {
        primary_release(dev)
    }

    fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) {
        drv()?.device(dev)?;
        Ok(())
    }

    fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) {
        let d = drv()?.device(dev)?;
        *out!(flags) = d.primary_flags.load(Ordering::Relaxed);
        *out!(active) = d.primary.lock().unwrap().as_ref().map_or(0, |p| (p.1 > 0) as c_int);
        Ok(())
    }

    fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) {
        drv()?.device(dev)?.primary_flags.store(flags, Ordering::Relaxed);
        Ok(())
    }

    // ---------------------------------------------------------------- context
    fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) {
        drv()?.device(dev)?;
        let c = Box::into_raw(Context::new(dev as usize, flags, false)?);
        CTX_STACK.with(|s| s.borrow_mut().push(c));
        *out!(ctx) = c as CUcontext;
        Ok(())
    }

    fn cuCtxDestroy_v2(ctx: CUcontext) {
        let c = ctx_ptr(ctx)?;
        if c.primary {
            return Err(CUDA_ERROR_INVALID_CONTEXT);
        }
        let _ = c.finish();
        CTX_STACK.with(|s| s.borrow_mut().retain(|&x| x != c as *const Context));
        let d = drv()?;
        (d.cl.clReleaseCommandQueue)(c.queue);
        drop(Box::from_raw(ctx as *mut Context));
        Ok(())
    }

    fn cuCtxPushCurrent_v2(ctx: CUcontext) {
        ctx_ptr(ctx)?;
        CTX_STACK.with(|s| s.borrow_mut().push(ctx as *const Context));
        Ok(())
    }

    fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) {
        let c = CTX_STACK.with(|s| s.borrow_mut().pop()).ok_or(CUDA_ERROR_INVALID_CONTEXT)?;
        if !ctx.is_null() {
            *ctx = c as CUcontext;
        }
        Ok(())
    }

    fn cuCtxSetCurrent(ctx: CUcontext) {
        set_current(ctx as *const Context);
        Ok(())
    }

    fn cuCtxGetCurrent(ctx: *mut CUcontext) {
        *out!(ctx) = CTX_STACK.with(|s| s.borrow().last().copied()).unwrap_or(std::ptr::null()) as CUcontext;
        Ok(())
    }

    fn cuCtxGetDevice(dev: *mut CUdevice) {
        *out!(dev) = current()?.dev as CUdevice;
        Ok(())
    }

    fn cuCtxGetFlags(flags: *mut c_uint) {
        *out!(flags) = current()?.flags;
        Ok(())
    }

    fn cuCtxGetId(ctx: CUcontext, id: *mut u64) {
        let c = if ctx.is_null() { current()? } else { ctx_ptr(ctx)? };
        *out!(id) = c.id;
        Ok(())
    }

    fn cuCtxSynchronize() {
        current()?.finish()
    }

    fn cuCtxGetApiVersion(_ctx: CUcontext, v: *mut c_uint) {
        *out!(v) = 3020;
        Ok(())
    }

    fn cuCtxSetLimit(limit: c_int, value: usize) {
        current()?.limits.lock().unwrap().insert(limit as u32, value);
        Ok(())
    }

    fn cuCtxGetLimit(value: *mut usize, limit: c_int) {
        let c = current()?;
        let default = match limit {
            0 => 1024,          // stack size
            1 => 1 << 20,       // printf fifo
            2 => 8 << 20,       // malloc heap
            _ => 0,
        };
        *out!(value) = c.limits.lock().unwrap().get(&(limit as u32)).copied().unwrap_or(default);
        Ok(())
    }

    fn cuCtxGetCacheConfig(cfg: *mut c_int) {
        *out!(cfg) = 0;
        Ok(())
    }

    fn cuCtxSetCacheConfig(_cfg: c_int) {
        Ok(())
    }

    fn cuCtxGetSharedMemConfig(cfg: *mut c_int) {
        *out!(cfg) = 1;
        Ok(())
    }

    fn cuCtxSetSharedMemConfig(_cfg: c_int) {
        Ok(())
    }

    fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) {
        if !least.is_null() {
            *least = 0;
        }
        if !greatest.is_null() {
            *greatest = 0;
        }
        Ok(())
    }

    // ----------------------------------------------------------------- module
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) {
        load_module(m, image)
    }

    fn cuModuleLoadDataEx(m: *mut CUmodule, image: *const c_void, _n: c_uint, _opts: *mut c_int, _vals: *mut *mut c_void) {
        load_module(m, image)
    }

    fn cuModuleLoadFatBinary(m: *mut CUmodule, image: *const c_void) {
        load_module(m, image)
    }

    fn cuModuleLoad(m: *mut CUmodule, fname: *const c_char) {
        let path = CStr::from_ptr(fname).to_string_lossy().to_string();
        let mut data = std::fs::read(&path).map_err(|_| CUDA_ERROR_NOT_FOUND)?;
        data.push(0);
        load_module(m, data.as_ptr() as *const c_void)
    }

    fn cuModuleUnload(m: CUmodule) {
        module(m)?;
        unload(Box::from_raw(m as *mut Module))
    }

    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) {
        let m = module(m)?;
        let n = CStr::from_ptr(name).to_string_lossy();
        let func = m.functions.get(n.as_ref()).ok_or_else(|| {
            // Internal CUDA device-runtime helper kernels are probed lazily and
            // usually never launched; log them only at higher verbosity.
            if !n.starts_with("__nv_static_") {
                driver::log!("kernel {n} not found");
            }
            CUDA_ERROR_NOT_FOUND
        })?;
        *out!(f) = &**func as *const Function as CUfunction;
        Ok(())
    }

    fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, m: CUmodule, name: *const c_char) {
        let m = module(m)?;
        let n = CStr::from_ptr(name).to_string_lossy();
        let &(a, s) = m.globals.get(n.as_ref()).ok_or(CUDA_ERROR_NOT_FOUND)?;
        if !dptr.is_null() {
            *dptr = a;
        }
        if !bytes.is_null() {
            *bytes = s as usize;
        }
        Ok(())
    }

    fn cuModuleGetLoadingMode(mode: *mut c_int) {
        *out!(mode) = 1; // CU_MODULE_EAGER_LOADING
        Ok(())
    }

    // ------------------------------------------------------- library (12.0+)
    fn cuLibraryLoadData(lib: *mut CUlibrary, code: *const c_void, _jo: *mut c_int, _jv: *mut *mut c_void, _nj: c_uint,
                         _lo: *mut c_int, _lv: *mut *mut c_void, _nl: c_uint) {
        let ctx = current()?;
        let m = load_image(ctx.dev, code)?;
        *out!(lib) = Box::into_raw(m) as CUlibrary;
        Ok(())
    }

    fn cuLibraryUnload(lib: CUlibrary) {
        module(lib)?;
        unload(Box::from_raw(lib as *mut Module))
    }

    fn cuLibraryGetModule(m: *mut CUmodule, lib: CUlibrary) {
        module(lib)?;
        *out!(m) = lib;
        Ok(())
    }

    fn cuLibraryGetKernel(k: *mut CUkernel, lib: CUlibrary, name: *const c_char) {
        cuModuleGetFunction(k, lib, name).into_res()
    }

    fn cuKernelGetFunction(f: *mut CUfunction, k: CUkernel) {
        func(k)?;
        *out!(f) = k;
        Ok(())
    }

    fn cuLibraryGetGlobal(dptr: *mut CUdeviceptr, bytes: *mut usize, lib: CUlibrary, name: *const c_char) {
        cuModuleGetGlobal_v2(dptr, bytes, lib, name).into_res()
    }

    // --------------------------------------------------------------- function
    fn cuFuncGetAttribute(v: *mut c_int, a: c_int, f: CUfunction) {
        let f = func(f)?;
        let dev = drv()?.devices[(*f.module).dev].max_wg;
        *out!(v) = match a {
            0 => f.max_threads.min(dev).min(1024) as c_int,          // MAX_THREADS_PER_BLOCK
            1 => f.info.static_shared as c_int,                      // SHARED_SIZE_BYTES
            2 => 0,                                                  // CONST_SIZE_BYTES
            3 => f.private_mem as c_int,                             // LOCAL_SIZE_BYTES
            4 => 32,                                                 // NUM_REGS
            5 => (drv()?.cc.0 * 10 + drv()?.cc.1) as c_int,          // PTX_VERSION
            6 => (drv()?.cc.0 * 10 + drv()?.cc.1) as c_int,          // BINARY_VERSION
            7 => 0,                                                  // CACHE_MODE_CA
            8 => f.attrs.lock().unwrap().get(&8).copied().unwrap_or(drv()?.devices[(*f.module).dev].local_mem as i32 - f.info.static_shared as i32),
            _ => f.attrs.lock().unwrap().get(&a).copied().unwrap_or(0),
        };
        Ok(())
    }

    fn cuFuncSetAttribute(f: CUfunction, a: c_int, v: c_int) {
        func(f)?.attrs.lock().unwrap().insert(a, v);
        Ok(())
    }

    fn cuFuncSetCacheConfig(f: CUfunction, _c: c_int) {
        func(f)?;
        Ok(())
    }

    fn cuFuncSetSharedMemConfig(f: CUfunction, _c: c_int) {
        func(f)?;
        Ok(())
    }

    fn cuFuncGetModule(m: *mut CUmodule, f: CUfunction) {
        *out!(m) = func(f)?.module as CUmodule;
        Ok(())
    }

    fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
                      shared: c_uint, stream: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void) {
        launch(func(f)?, [gx, gy, gz], [bx, by, bz], shared, stream, params, extra)
    }

    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(n: *mut c_int, f: CUfunction, block: c_int, _smem: usize) {
        let f = func(f)?;
        *out!(n) = (f.max_threads.max(1) as c_int / block.max(1)).max(1);
        Ok(())
    }

    fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(n: *mut c_int, f: CUfunction, block: c_int, _smem: usize, _fl: c_uint) {
        let f = func(f)?;
        *out!(n) = (f.max_threads.max(1) as c_int / block.max(1)).max(1);
        Ok(())
    }

    // ----------------------------------------------------------------- memory
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, size: usize) {
        *out!(p) = mem_alloc(size, AllocKind::Device)?;
        Ok(())
    }

    fn cuMemAllocPitch_v2(p: *mut CUdeviceptr, pitch: *mut usize, width: usize, height: usize, _elem: c_uint) {
        let pt = (width + 255) / 256 * 256;
        *out!(pitch) = pt;
        *out!(p) = mem_alloc(pt * height, AllocKind::Device)?;
        Ok(())
    }

    fn cuMemAllocManaged(p: *mut CUdeviceptr, size: usize, _flags: c_uint) {
        *out!(p) = mem_alloc(size, AllocKind::Shared)?;
        Ok(())
    }

    fn cuMemAllocAsync(p: *mut CUdeviceptr, size: usize, _s: CUstream) {
        *out!(p) = mem_alloc(size, AllocKind::Device)?;
        Ok(())
    }

    fn cuMemFree_v2(p: CUdeviceptr) {
        mem_free(p)
    }

    fn cuMemFreeAsync(p: CUdeviceptr, _s: CUstream) {
        mem_free(p)
    }

    fn cuMemAllocHost_v2(p: *mut *mut c_void, size: usize) {
        *out!(p) = mem_alloc(size, AllocKind::Host)? as *mut c_void;
        Ok(())
    }

    fn cuMemHostAlloc(p: *mut *mut c_void, size: usize, _flags: c_uint) {
        *out!(p) = mem_alloc(size, AllocKind::Host)? as *mut c_void;
        Ok(())
    }

    fn cuMemFreeHost(p: *mut c_void) {
        mem_free(p as u64)
    }

    fn cuMemHostGetDevicePointer_v2(d: *mut CUdeviceptr, p: *mut c_void, _flags: c_uint) {
        *out!(d) = p as u64;
        Ok(())
    }

    fn cuMemHostGetFlags(flags: *mut c_uint, _p: *mut c_void) {
        *out!(flags) = 0;
        Ok(())
    }

    fn cuMemHostRegister_v2(_p: *mut c_void, _size: usize, _flags: c_uint) {
        Ok(())
    }

    fn cuMemHostUnregister(_p: *mut c_void) {
        Ok(())
    }

    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) {
        let dev = current()?.device();
        let used: u64 = drv()?.allocs.lock().unwrap().values().filter(|a| a.kind != AllocKind::Host).map(|a| a.size).sum();
        *out!(total) = dev.total_mem as usize;
        *out!(free) = dev.total_mem.saturating_sub(used) as usize;
        Ok(())
    }

    fn cuMemGetAddressRange_v2(base: *mut CUdeviceptr, size: *mut usize, p: CUdeviceptr) {
        let (b, a) = find_alloc(p).ok_or(CUDA_ERROR_NOT_FOUND)?;
        if !base.is_null() {
            *base = b;
        }
        if !size.is_null() {
            *size = a.size as usize;
        }
        Ok(())
    }

    fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, n: usize) {
        memcpy(dst, src, n, current()?.queue, true)
    }

    fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) {
        memcpy(dst, src, n, queue_of(s)?, false)
    }

    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) {
        memcpy(dst, src as u64, n, current()?.queue, true)
    }

    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) {
        memcpy(dst as u64, src, n, current()?.queue, true)
    }

    fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize) {
        memcpy(dst, src, n, current()?.queue, false)
    }

    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: CUstream) {
        memcpy(dst, src as u64, n, queue_of(s)?, false)
    }

    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) {
        // Pageable destinations must be complete on return.
        let blocking = find_alloc(dst as u64).is_none();
        memcpy(dst as u64, src, n, queue_of(s)?, blocking)
    }

    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) {
        memcpy(dst, src, n, queue_of(s)?, false)
    }

    fn cuMemcpy2D_v2(p: *const CUDA_MEMCPY2D) {
        memcpy2d(&*p, null_mut(), true)
    }

    fn cuMemcpy2DUnaligned_v2(p: *const CUDA_MEMCPY2D) {
        memcpy2d(&*p, null_mut(), true)
    }

    fn cuMemcpy2DAsync_v2(p: *const CUDA_MEMCPY2D, s: CUstream) {
        memcpy2d(&*p, s, false)
    }

    fn cuMemsetD8_v2(dst: CUdeviceptr, v: u8, n: usize) {
        memset(dst, &[v], n, current()?.queue, true)
    }

    fn cuMemsetD16_v2(dst: CUdeviceptr, v: u16, n: usize) {
        memset(dst, &v.to_le_bytes(), n, current()?.queue, true)
    }

    fn cuMemsetD32_v2(dst: CUdeviceptr, v: u32, n: usize) {
        memset(dst, &v.to_le_bytes(), n, current()?.queue, true)
    }

    fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: CUstream) {
        memset(dst, &[v], n, queue_of(s)?, false)
    }

    fn cuMemsetD16Async(dst: CUdeviceptr, v: u16, n: usize, s: CUstream) {
        memset(dst, &v.to_le_bytes(), n, queue_of(s)?, false)
    }

    fn cuMemsetD32Async(dst: CUdeviceptr, v: u32, n: usize, s: CUstream) {
        memset(dst, &v.to_le_bytes(), n, queue_of(s)?, false)
    }

    fn cuMemsetD2D8_v2(dst: CUdeviceptr, pitch: usize, v: u8, w: usize, h: usize) {
        let q = current()?.queue;
        for r in 0..h {
            memset(dst + (r * pitch) as u64, &[v], w, q, false)?;
        }
        current()?.finish()
    }

    fn cuMemsetD2D32_v2(dst: CUdeviceptr, pitch: usize, v: u32, w: usize, h: usize) {
        let q = current()?.queue;
        for r in 0..h {
            memset(dst + (r * pitch) as u64, &v.to_le_bytes(), w, q, false)?;
        }
        current()?.finish()
    }

    fn cuPointerGetAttribute(data: *mut c_void, a: c_int, p: CUdeviceptr) {
        let found = find_alloc(p);
        match a {
            1 => *(data as *mut CUcontext) = current().map(|c| c as *const Context as CUcontext).unwrap_or(null_mut()),
            2 => {
                // MEMORY_TYPE: host 1, device 2
                let (_, al) = found.ok_or(CUDA_ERROR_INVALID_VALUE)?;
                *(data as *mut c_uint) = if al.kind == AllocKind::Host { 1 } else { 2 };
            }
            3 | 4 => *(data as *mut u64) = p, // DEVICE_POINTER / HOST_POINTER
            6 => *(data as *mut c_int) = 1,   // SYNC_MEMOPS
            7 => *(data as *mut u64) = found.map_or(0, |f| f.0), // BUFFER_ID
            8 => *(data as *mut c_int) = found.map_or(0, |f| (f.1.kind == AllocKind::Shared) as c_int), // IS_MANAGED
            9 => *(data as *mut c_int) = found.map_or(0, |f| f.1.dev as c_int), // DEVICE_ORDINAL
            12 => *(data as *mut u64) = found.map_or(0, |f| f.0),               // RANGE_START_ADDR
            13 => *(data as *mut usize) = found.map_or(0, |f| f.1.size as usize), // RANGE_SIZE
            14 => *(data as *mut c_int) = found.is_some() as c_int,             // MAPPED
            _ => return Err(CUDA_ERROR_INVALID_VALUE),
        }
        Ok(())
    }

    fn cuPointerGetAttributes(n: c_uint, attrs: *const c_int, data: *mut *mut c_void, p: CUdeviceptr) {
        for i in 0..n as usize {
            let a = *attrs.add(i);
            let r = cuPointerGetAttribute(*data.add(i), a, p);
            if r != 0 && a == 2 {
                *(*data.add(i) as *mut c_uint) = 0; // unregistered host memory
            }
        }
        Ok(())
    }

    fn cuPointerSetAttribute(_v: *const c_void, _a: c_int, _p: CUdeviceptr) {
        Ok(())
    }

    // ---------------------------------------------------------------- streams
    fn cuStreamCreate(s: *mut CUstream, flags: c_uint) {
        let ctx = current()?;
        *out!(s) = Box::into_raw(Box::new(Stream { ctx, flags, priority: 0 })) as CUstream;
        Ok(())
    }

    fn cuStreamCreateWithPriority(s: *mut CUstream, flags: c_uint, priority: c_int) {
        let ctx = current()?;
        *out!(s) = Box::into_raw(Box::new(Stream { ctx, flags, priority })) as CUstream;
        Ok(())
    }

    fn cuStreamDestroy_v2(s: CUstream) {
        if s.is_null() {
            return Err(CUDA_ERROR_INVALID_HANDLE);
        }
        drop(Box::from_raw(s as *mut Stream));
        Ok(())
    }

    fn cuStreamSynchronize(s: CUstream) {
        let q = queue_of(s)?;
        cl_check((drv()?.cl.clFinish)(q), "clFinish")
    }

    fn cuStreamQuery(s: CUstream) {
        let q = queue_of(s)?;
        cl_check((drv()?.cl.clFinish)(q), "clFinish")
    }

    fn cuStreamWaitEvent(_s: CUstream, _e: CUevent, _flags: c_uint) {
        Ok(()) // single in-order queue per context
    }

    fn cuStreamGetPriority(s: CUstream, p: *mut c_int) {
        *out!(p) = if s.is_null() { 0 } else { (*(s as *const Stream)).priority };
        Ok(())
    }

    fn cuStreamGetFlags(s: CUstream, f: *mut c_uint) {
        *out!(f) = if s.is_null() { 0 } else { (*(s as *const Stream)).flags };
        Ok(())
    }

    fn cuStreamGetCtx(s: CUstream, c: *mut CUcontext) {
        *out!(c) = if s.is_null() || s as usize <= 2 {
            current()? as *const Context as CUcontext
        } else {
            (*(s as *const Stream)).ctx as CUcontext
        };
        Ok(())
    }

    fn cuStreamIsCapturing(_s: CUstream, st: *mut c_int) {
        *out!(st) = 0;
        Ok(())
    }

    fn cuStreamGetCaptureInfo_v2(_s: CUstream, st: *mut c_int, _id: *mut u64, _g: *mut *mut c_void,
                                 _deps: *mut *const c_void, _n: *mut usize) {
        *out!(st) = 0;
        Ok(())
    }

    fn cuLaunchHostFunc(s: CUstream, f: Option<unsafe extern "C" fn(*mut c_void)>, data: *mut c_void) {
        let q = queue_of(s)?;
        cl_check((drv()?.cl.clFinish)(q), "clFinish")?;
        if let Some(f) = f {
            f(data);
        }
        Ok(())
    }

    fn cuStreamAddCallback(s: CUstream, f: Option<unsafe extern "C" fn(CUstream, CUresult, *mut c_void)>, data: *mut c_void, _fl: c_uint) {
        let q = queue_of(s)?;
        cl_check((drv()?.cl.clFinish)(q), "clFinish")?;
        if let Some(f) = f {
            f(s, CUDA_SUCCESS, data);
        }
        Ok(())
    }

    // ----------------------------------------------------------------- events
    fn cuEventCreate(e: *mut CUevent, flags: c_uint) {
        *out!(e) = Box::into_raw(Box::new(Event { ev: std::sync::Mutex::new(null_mut()), flags })) as CUevent;
        Ok(())
    }

    fn cuEventDestroy_v2(e: CUevent) {
        if e.is_null() {
            return Err(CUDA_ERROR_INVALID_HANDLE);
        }
        let ev = Box::from_raw(e as *mut Event);
        let x = *ev.ev.lock().unwrap();
        if !x.is_null() {
            (drv()?.cl.clReleaseEvent)(x);
        }
        Ok(())
    }

    fn cuEventRecord(e: CUevent, s: CUstream) {
        record(e, s)
    }

    fn cuEventRecordWithFlags(e: CUevent, s: CUstream, _f: c_uint) {
        record(e, s)
    }

    fn cuEventSynchronize(e: CUevent) {
        let ev = event(e)?;
        if !ev.is_null() {
            cl_check((drv()?.cl.clWaitForEvents)(1, &ev), "clWaitForEvents")?;
        }
        Ok(())
    }

    fn cuEventQuery(e: CUevent) {
        let ev = event(e)?;
        if ev.is_null() {
            return Ok(());
        }
        let mut st: cl_int = 0;
        (drv()?.cl.clGetEventInfo)(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, 4, &mut st as *mut i32 as *mut c_void, null_mut());
        if st == CL_COMPLETE { Ok(()) } else { Err(CUDA_ERROR_NOT_READY) }
    }

    fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) {
        let (a, b) = (event(start)?, event(end)?);
        if a.is_null() || b.is_null() {
            return Err(CUDA_ERROR_INVALID_HANDLE);
        }
        let d = drv()?;
        cl_check((d.cl.clWaitForEvents)(1, &b), "clWaitForEvents")?;
        let t = |e: cl_event| {
            let mut v = 0u64;
            (d.cl.clGetEventProfilingInfo)(e, CL_PROFILING_COMMAND_END, 8, &mut v as *mut u64 as *mut c_void, null_mut());
            v
        };
        *out!(ms) = (t(b) as f64 - t(a) as f64) as f32 / 1e6;
        Ok(())
    }

    // ------------------------------------------------------------------ misc
    fn cuGetErrorName(e: CUresult, s: *mut *const c_char) {
        *out!(s) = static_cstr(error_name(e));
        Ok(())
    }

    fn cuGetErrorString(e: CUresult, s: *mut *const c_char) {
        *out!(s) = static_cstr(error_string(e));
        Ok(())
    }

    fn cuProfilerStart() {
        Ok(())
    }

    fn cuProfilerStop() {
        Ok(())
    }

    fn cuGetProcAddress(sym: *const c_char, pfn: *mut *mut c_void, ver: c_int, flags: u64) {
        get_proc(sym, pfn, ver, flags)
    }

    fn cuGetProcAddress_v2(sym: *const c_char, pfn: *mut *mut c_void, ver: c_int, flags: u64, status: *mut c_int) {
        let r = get_proc(sym, pfn, ver, flags);
        if !status.is_null() {
            *status = if r.is_ok() { 0 } else { 1 }; // CU_GET_PROC_ADDRESS_SUCCESS / SYMBOL_NOT_FOUND
        }
        r
    }

    fn cuGetExportTable(table: *mut *const c_void, id: *const CUuuid) {
        crate::dark::export_table(table, id)
    }
}

trait IntoRes {
    fn into_res(self) -> Res;
}
impl IntoRes for CUresult {
    fn into_res(self) -> Res {
        if self == 0 { Ok(()) } else { Err(self) }
    }
}

fn static_cstr(s: &'static str) -> *const c_char {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<&'static str, std::ffi::CString>>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap();
    let m = c.get_or_insert_with(HashMap::new);
    m.entry(s).or_insert_with(|| std::ffi::CString::new(s).unwrap()).as_ptr()
}

fn event(e: CUevent) -> Res<cl_event> {
    if e.is_null() {
        return Err(CUDA_ERROR_INVALID_HANDLE);
    }
    Ok(*unsafe { &*(e as *const Event) }.ev.lock().unwrap())
}

fn record(e: CUevent, s: CUstream) -> Res {
    if e.is_null() {
        return Err(CUDA_ERROR_INVALID_HANDLE);
    }
    let d = drv()?;
    let ev = unsafe { &*(e as *const Event) };
    let q = queue_of(s)?;
    let mut new = null_mut();
    cl_check(unsafe { (d.cl.clEnqueueMarkerWithWaitList)(q, 0, std::ptr::null(), &mut new) }, "clEnqueueMarker")?;
    unsafe { (d.cl.clFlush)(q) };
    let old = std::mem::replace(&mut *ev.ev.lock().unwrap(), new);
    if !old.is_null() {
        unsafe { (d.cl.clReleaseEvent)(old) };
    }
    Ok(())
}

/// Resolve a driver entry point by its unversioned name, picking the
/// versioned variant matching the requested CUDA version.
unsafe fn get_proc(sym: *const c_char, pfn: *mut *mut c_void, ver: c_int, _flags: u64) -> Res {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static TABLE: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let t = TABLE.get_or_init(|| table().into_iter().collect());
    let name = CStr::from_ptr(sym).to_string_lossy();
    if pfn.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let mut cands: Vec<String> = Vec::new();
    // _v2 variants appeared with CUDA 3.2 (pointer-sized types) or later API revisions.
    if ver >= 3020 {
        cands.push(format!("{name}_v2"));
    }
    cands.push(name.to_string());
    for c in &cands {
        if let Some(&a) = t.get(c.as_str()) {
            *pfn = a as *mut c_void;
            return Ok(());
        }
    }
    *pfn = null_mut();
    driver::log!("cuGetProcAddress: {name} (v{ver}) not implemented");
    Err(CUDA_ERROR_NOT_FOUND)
}
