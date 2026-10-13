//! Driver state: devices, contexts, modules, memory — all backed by OpenCL + USM.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_void, CString};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::cl::*;
use crate::types::*;

pub type Res<T = ()> = Result<T, CUresult>;

pub fn log_enabled() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| std::env::var("IGNUDA_LOG").map_or(false, |v| v != "0"))
}

macro_rules! log {
    ($($t:tt)*) => { if $crate::driver::log_enabled() { eprintln!("[ignuda] {}", format!($($t)*)); } };
}
pub(crate) use log;

pub struct Device {
    pub platform: cl_platform_id,
    pub id: cl_device_id,
    pub name: String,
    pub is_gpu: bool,
    pub total_mem: u64,
    pub compute_units: u32,
    pub max_wg: usize,
    pub local_mem: u64,
    pub clock_mhz: u32,
    pub const_mem: u64,
    pub l2: u64,
    pub fp64: bool,
    pub float_atomics: bool,
    pub uuid: [u8; 16],
    pub pci: [u32; 4],
    pub cl_ctx: cl_context,
    pub usm: Usm,
    pub primary: Mutex<Option<(Box<Context>, u32)>>,
    pub primary_flags: AtomicU32,
}

pub struct Driver {
    pub cl: Cl,
    pub devices: Vec<Device>,
    pub cc: (i32, i32),
    pub allocs: Mutex<BTreeMap<u64, Alloc>>,
    /// Device addresses of module globals (not USM allocations).
    pub globals: Mutex<BTreeMap<u64, u64>>,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocKind {
    Device,
    Host,
    Shared,
}

#[derive(Clone, Copy, Debug)]
pub struct Alloc {
    pub size: u64,
    pub kind: AllocKind,
    pub dev: usize,
}

pub struct Context {
    pub dev: usize,
    pub queue: cl_command_queue,
    pub flags: u32,
    pub primary: bool,
    pub id: u64,
    pub limits: Mutex<HashMap<u32, usize>>,
    /// Helper module with copy kernels, built on first use.
    pub helpers: OnceLock<Box<Module>>,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

pub struct Module {
    pub dev: usize,
    pub program: cl_program,
    pub functions: HashMap<String, Box<Function>>,
    pub globals: HashMap<String, (u64, u64)>,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

pub struct Function {
    pub kernel: Mutex<cl_kernel>,
    pub info: ptx2spirv::KernelInfo,
    pub module: *const Module,
    pub max_threads: usize,
    pub local_mem: u64,
    pub private_mem: u64,
    pub attrs: Mutex<HashMap<i32, i32>>,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

pub struct Stream {
    pub ctx: *const Context,
    pub flags: u32,
    pub priority: i32,
}

pub struct Event {
    pub ev: Mutex<cl_event>,
    pub flags: u32,
}

static DRIVER: OnceLock<Result<Driver, CUresult>> = OnceLock::new();
static CTX_IDS: AtomicU32 = AtomicU32::new(1);

thread_local! {
    pub static CTX_STACK: RefCell<Vec<*const Context>> = const { RefCell::new(Vec::new()) };
}

pub fn init() -> Res {
    DRIVER.get_or_init(|| Driver::load().map_err(|e| {
        log!("initialization failed: {e}");
        CUDA_ERROR_NO_DEVICE
    }))
    .as_ref()
    .map(|_| ())
    .map_err(|e| *e)
}

pub fn drv() -> Res<&'static Driver> {
    match DRIVER.get() {
        Some(Ok(d)) => Ok(d),
        Some(Err(_)) => Err(CUDA_ERROR_NO_DEVICE),
        None => Err(CUDA_ERROR_NOT_INITIALIZED),
    }
}

pub fn cl_check(r: cl_int, what: &str) -> Res {
    if r == CL_SUCCESS {
        Ok(())
    } else {
        log!("{what} failed with OpenCL error {r}");
        Err(match r {
            -4 | -5 | -6 => CUDA_ERROR_OUT_OF_MEMORY,
            -54 | -55 => CUDA_ERROR_INVALID_VALUE,
            _ => CUDA_ERROR_UNKNOWN,
        })
    }
}

impl Driver {
    fn load() -> Result<Driver, String> {
        let cl = Cl::load()?;
        let want = std::env::var("IGNUDA_DEVICE_TYPE").unwrap_or_default();
        let mut devices = Vec::new();
        unsafe {
            let mut n = 0u32;
            (cl.clGetPlatformIDs)(0, null_mut(), &mut n);
            let mut plats = vec![null_mut(); n as usize];
            (cl.clGetPlatformIDs)(n, plats.as_mut_ptr(), null_mut());
            let mut cands: Vec<(cl_platform_id, cl_device_id, bool)> = Vec::new();
            for &p in &plats {
                let mut nd = 0u32;
                if (cl.clGetDeviceIDs)(p, CL_DEVICE_TYPE_ALL, 0, null_mut(), &mut nd) != 0 {
                    continue;
                }
                let mut ds = vec![null_mut(); nd as usize];
                (cl.clGetDeviceIDs)(p, CL_DEVICE_TYPE_ALL, nd, ds.as_mut_ptr(), null_mut());
                for d in ds {
                    let ty: u64 = cl.device_info(d, CL_DEVICE_TYPE);
                    let exts = cl.device_string(d, CL_DEVICE_EXTENSIONS);
                    if !exts.contains("cl_intel_unified_shared_memory") || !exts.contains("cl_khr_il_program") {
                        log!("skipping {} (needs USM and SPIR-V support)", cl.device_string(d, CL_DEVICE_NAME));
                        continue;
                    }
                    cands.push((p, d, ty & CL_DEVICE_TYPE_GPU != 0));
                }
            }
            let any_gpu = cands.iter().any(|c| c.2);
            for (p, d, gpu) in cands {
                let take = match want.as_str() {
                    "cpu" => !gpu,
                    "gpu" => gpu,
                    "all" => true,
                    _ => gpu || !any_gpu,
                };
                if !take {
                    continue;
                }
                let usm = match cl.usm(p) {
                    Some(u) => u,
                    None => continue,
                };
                let props = [CL_CONTEXT_PLATFORM, p as isize, 0];
                let mut err = 0;
                let ctx = (cl.clCreateContext)(props.as_ptr(), 1, &d, null(), null_mut(), &mut err);
                if err != 0 {
                    log!("clCreateContext failed: {err}");
                    continue;
                }
                let exts = cl.device_string(d, CL_DEVICE_EXTENSIONS);
                let fp_caps: u64 = if exts.contains("cl_ext_float_atomics") {
                    cl.device_info(d, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT)
                } else {
                    0
                };
                let mut uuid = [0u8; 16];
                let ub = cl.device_bytes(d, CL_DEVICE_UUID_KHR);
                if ub.len() == 16 {
                    uuid.copy_from_slice(&ub);
                }
                let pb = cl.device_bytes(d, CL_DEVICE_PCI_BUS_INFO_KHR);
                let mut pci = [0u32; 4];
                for (k, c) in pb.chunks(4).take(4).enumerate() {
                    pci[k] = u32::from_le_bytes(c.try_into().unwrap_or([0; 4]));
                }
                let dev = Device {
                    platform: p,
                    id: d,
                    name: cl.device_string(d, CL_DEVICE_NAME),
                    is_gpu: gpu,
                    total_mem: cl.device_info(d, CL_DEVICE_GLOBAL_MEM_SIZE),
                    compute_units: cl.device_info(d, CL_DEVICE_MAX_COMPUTE_UNITS),
                    max_wg: cl.device_info(d, CL_DEVICE_MAX_WORK_GROUP_SIZE),
                    local_mem: cl.device_info(d, CL_DEVICE_LOCAL_MEM_SIZE),
                    clock_mhz: cl.device_info(d, CL_DEVICE_MAX_CLOCK_FREQUENCY),
                    const_mem: cl.device_info(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE),
                    l2: cl.device_info(d, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE),
                    fp64: cl.device_info::<u64>(d, CL_DEVICE_DOUBLE_FP_CONFIG) != 0,
                    float_atomics: fp_caps & (1 << 1) != 0,
                    uuid,
                    pci,
                    cl_ctx: ctx,
                    usm,
                    primary: Mutex::new(None),
                    primary_flags: AtomicU32::new(0),
                };
                log!("device {}: {} ({} CUs, {} MiB, fp64={}, gpu={})", devices.len(), dev.name, dev.compute_units, dev.total_mem >> 20, dev.fp64, gpu);
                devices.push(dev);
            }
        }
        if devices.is_empty() {
            return Err("no usable OpenCL device".into());
        }
        let cc = std::env::var("IGNUDA_CC")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once('.')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((8, 6));
        Ok(Driver { cl, devices, cc, allocs: Mutex::new(BTreeMap::new()), globals: Mutex::new(BTreeMap::new()) })
    }

    pub fn device(&self, d: CUdevice) -> Res<&Device> {
        self.devices.get(d as usize).ok_or(CUDA_ERROR_INVALID_DEVICE)
    }
}

// ------------------------------------------------------------------ contexts

impl Context {
    pub fn new(dev: usize, flags: u32, primary: bool) -> Res<Box<Context>> {
        let d = drv()?;
        let device = &d.devices[dev];
        let props = [CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0];
        let mut err = 0;
        let q = unsafe { (d.cl.clCreateCommandQueueWithProperties)(device.cl_ctx, device.id, props.as_ptr(), &mut err) };
        cl_check(err, "clCreateCommandQueueWithProperties")?;
        Ok(Box::new(Context {
            dev,
            queue: q,
            flags,
            primary,
            id: CTX_IDS.fetch_add(1, Ordering::Relaxed) as u64,
            limits: Mutex::new(HashMap::new()),
            helpers: OnceLock::new(),
        }))
    }

    pub fn device(&self) -> &'static Device {
        &drv().unwrap().devices[self.dev]
    }

    pub fn finish(&self) -> Res {
        let d = drv()?;
        cl_check(unsafe { (d.cl.clFinish)(self.queue) }, "clFinish")
    }
}

pub fn current() -> Res<&'static Context> {
    let c = CTX_STACK.with(|s| s.borrow().last().copied());
    match c {
        Some(c) if !c.is_null() => Ok(unsafe { &*c }),
        _ => Err(CUDA_ERROR_INVALID_CONTEXT),
    }
}

pub fn set_current(c: *const Context) {
    CTX_STACK.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(top) = s.last_mut() {
            *top = c;
        } else if !c.is_null() {
            s.push(c);
        }
    });
}

pub fn primary_retain(dev: CUdevice) -> Res<*const Context> {
    let d = drv()?;
    let device = d.device(dev)?;
    let mut p = device.primary.lock().unwrap();
    match p.as_mut() {
        Some((c, rc)) => {
            *rc += 1;
            Ok(&**c as *const Context)
        }
        None => {
            let c = Context::new(dev as usize, device.primary_flags.load(Ordering::Relaxed), true)?;
            let ptr = &*c as *const Context;
            *p = Some((c, 1));
            Ok(ptr)
        }
    }
}

/// Primary contexts are never actually destroyed (queues are cheap and other
/// threads may still hold the handle); release only drops the refcount.
pub fn primary_release(dev: CUdevice) -> Res {
    let d = drv()?;
    let device = d.device(dev)?;
    let mut p = device.primary.lock().unwrap();
    if let Some((_, rc)) = p.as_mut() {
        *rc = rc.saturating_sub(1);
    }
    Ok(())
}

pub fn queue_of(stream: CUstream) -> Res<cl_command_queue> {
    // All streams of a context share its in-order queue for now (correct,
    // conservative ordering; no inter-stream concurrency).
    if stream.is_null() || stream as usize == 1 || stream as usize == 2 {
        return Ok(current()?.queue);
    }
    let s = unsafe { &*(stream as *const Stream) };
    Ok(unsafe { &*s.ctx }.queue)
}

// -------------------------------------------------------------------- memory

pub fn mem_alloc(size: usize, kind: AllocKind) -> Res<u64> {
    let d = drv()?;
    let ctx = current()?;
    let dev = ctx.device();
    let size_alloc = size.max(1);
    let mut err = 0;
    let p = unsafe {
        match kind {
            AllocKind::Device => (dev.usm.clDeviceMemAllocINTEL)(dev.cl_ctx, dev.id, null(), size_alloc, 0, &mut err),
            AllocKind::Host => (dev.usm.clHostMemAllocINTEL)(dev.cl_ctx, null(), size_alloc, 0, &mut err),
            AllocKind::Shared => (dev.usm.clSharedMemAllocINTEL)(dev.cl_ctx, dev.id, null(), size_alloc, 0, &mut err),
        }
    };
    if err != 0 || p.is_null() {
        log!("USM allocation of {size} bytes failed: {err}");
        return Err(CUDA_ERROR_OUT_OF_MEMORY);
    }
    d.allocs.lock().unwrap().insert(p as u64, Alloc { size: size as u64, kind, dev: ctx.dev });
    log!("alloc {kind:?} {size} -> {p:p}");
    Ok(p as u64)
}

pub fn mem_free(p: u64) -> Res {
    if p == 0 {
        return Ok(());
    }
    let d = drv()?;
    let a = d.allocs.lock().unwrap().remove(&p).ok_or(CUDA_ERROR_INVALID_VALUE)?;
    let dev = &d.devices[a.dev];
    cl_check(unsafe { (dev.usm.clMemBlockingFreeINTEL)(dev.cl_ctx, p as *mut c_void) }, "clMemBlockingFreeINTEL")
}

/// Find the allocation containing `p`: (base, alloc).
pub fn find_alloc(p: u64) -> Option<(u64, Alloc)> {
    let d = drv().ok()?;
    let m = d.allocs.lock().unwrap();
    let (&base, &a) = m.range(..=p).next_back()?;
    if p < base + a.size.max(1) {
        Some((base, a))
    } else {
        None
    }
}

fn is_module_global(p: u64) -> bool {
    let Ok(d) = drv() else { return false };
    let m = d.globals.lock().unwrap();
    m.range(..=p).next_back().map_or(false, |(&b, &s)| p < b + s.max(1))
}

/// Copy `n` bytes. Pointers may be USM allocations, plain host memory, or
/// module globals (which need a kernel to reach).
pub fn memcpy(dst: u64, src: u64, n: usize, q: cl_command_queue, blocking: bool) -> Res {
    if n == 0 {
        return Ok(());
    }
    let d = drv()?;
    let ctx = current()?;
    if is_module_global(dst) || is_module_global(src) {
        return copy_kernel(ctx, dst, src, n, q, blocking);
    }
    let dev = ctx.device();
    // Non-USM host memory may be reused by the caller right after return.
    let host_src = find_alloc(src).is_none();
    let blocking = blocking || host_src;
    cl_check(
        unsafe {
            (dev.usm.clEnqueueMemcpyINTEL)(q, blocking as u32, dst as *mut c_void, src as *const c_void, n, 0, null(), null_mut())
        },
        "clEnqueueMemcpyINTEL",
    )?;
    let _ = d;
    Ok(())
}

pub fn memset(dst: u64, pattern: &[u8], count: usize, q: cl_command_queue, blocking: bool) -> Res {
    if count == 0 {
        return Ok(());
    }
    let d = drv()?;
    let ctx = current()?;
    let dev = ctx.device();
    if find_alloc(dst).is_none() {
        // plain host memory
        unsafe {
            let p = dst as *mut u8;
            for k in 0..count {
                std::ptr::copy_nonoverlapping(pattern.as_ptr(), p.add(k * pattern.len()), pattern.len());
            }
        }
        return Ok(());
    }
    cl_check(
        unsafe {
            (dev.usm.clEnqueueMemFillINTEL)(
                q,
                dst as *mut c_void,
                pattern.as_ptr() as *const c_void,
                pattern.len(),
                count * pattern.len(),
                0,
                null(),
                null_mut(),
            )
        },
        "clEnqueueMemFillINTEL",
    )?;
    if blocking {
        cl_check(unsafe { (d.cl.clFinish)(q) }, "clFinish")?;
    }
    Ok(())
}

const HELPER_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry __ignuda_copy(.param .u64 dst, .param .u64 src, .param .u64 n)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [dst];
    ld.param.u64 %rd2, [src];
    ld.param.u64 %rd3, [n];
    mov.u64 %rd4, 0;
$L_loop:
    setp.ge.u64 %p1, %rd4, %rd3;
    @%p1 bra $L_done;
    add.s64 %rd5, %rd2, %rd4;
    ld.u8 %rs1, [%rd5];
    add.s64 %rd6, %rd1, %rd4;
    st.u8 [%rd6], %rs1;
    add.s64 %rd4, %rd4, 1;
    bra $L_loop;
$L_done:
    ret;
}
"#;

/// Byte copy through a kernel; used where one side is a module global.
/// Host-memory sides are staged through a USM host allocation.
fn copy_kernel(ctx: &Context, dst: u64, src: u64, n: usize, q: cl_command_queue, _blocking: bool) -> Res {
    let d = drv()?;
    let helpers = match ctx.helpers.get() {
        Some(h) => h,
        None => {
            let m = load_ptx(ctx.dev, HELPER_PTX)?;
            let _ = ctx.helpers.set(m);
            ctx.helpers.get().unwrap()
        }
    };
    let f = helpers.functions.get("__ignuda_copy").ok_or(CUDA_ERROR_NOT_FOUND)?;
    let device_side = |p: u64| is_module_global(p) || find_alloc(p).is_some();
    let mut staging = None;
    let (kd, ks) = (
        if device_side(dst) { dst } else {
            let s = mem_alloc(n, AllocKind::Host)?;
            staging = Some((s, dst, true));
            s
        },
        if device_side(src) { src } else {
            let s = mem_alloc(n, AllocKind::Host)?;
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, s as *mut u8, n) };
            staging = Some((s, 0, false));
            s
        },
    );
    {
        let k = f.kernel.lock().unwrap();
        for (i, v) in [kd, ks, n as u64].iter().enumerate() {
            cl_check(unsafe { (d.cl.clSetKernelArg)(*k, i as u32, 8, v as *const u64 as *const c_void) }, "clSetKernelArg")?;
        }
        let one = 1usize;
        cl_check(
            unsafe { (d.cl.clEnqueueNDRangeKernel)(q, *k, 1, null(), &one, &one, 0, null(), null_mut()) },
            "clEnqueueNDRangeKernel",
        )?;
    }
    cl_check(unsafe { (d.cl.clFinish)(q) }, "clFinish")?;
    if let Some((s, host, to_host)) = staging {
        if to_host {
            unsafe { std::ptr::copy_nonoverlapping(s as *const u8, host as *mut u8, n) };
        }
        mem_free(s)?;
    }
    Ok(())
}

// ------------------------------------------------------------------- modules

/// Load a module image: PTX text, a fatbin container, or a fatbin wrapper.
///
/// # Safety
/// `image` must point to a valid image as accepted by cuModuleLoadData.
pub unsafe fn load_image(dev: usize, image: *const c_void) -> Res<Box<Module>> {
    let p = image as *const u8;
    let magic = u32::from_le_bytes(std::slice::from_raw_parts(p, 4).try_into().unwrap());
    let fatbin_ptr = if magic == crate::fatbin::WRAPPER_MAGIC {
        // struct { int magic; int version; const void *data; void *filename; }
        *(p.add(8) as *const *const u8)
    } else {
        p
    };
    let magic2 = u32::from_le_bytes(std::slice::from_raw_parts(fatbin_ptr, 4).try_into().unwrap());
    if magic2 == crate::fatbin::FATBIN_MAGIC {
        let len = crate::fatbin::container_len(fatbin_ptr);
        let bytes = std::slice::from_raw_parts(fatbin_ptr, len);
        let entries = crate::fatbin::parse(bytes).map_err(|e| {
            log!("fatbin: {e}");
            CUDA_ERROR_INVALID_IMAGE
        })?;
        let (maj, min) = drv()?.cc;
        let arch = (maj * 10 + min) as u32;
        let e = crate::fatbin::best_ptx(&entries, arch).ok_or_else(|| {
            log!("fatbin has no PTX (only SASS for {:?}); rebuild with PTX included", entries.iter().map(|e| e.arch).collect::<Vec<_>>());
            CUDA_ERROR_NO_BINARY_FOR_GPU
        })?;
        let text = String::from_utf8_lossy(&e.data);
        let text = text.trim_end_matches('\0');
        return load_ptx(dev, text);
    }
    if &std::slice::from_raw_parts(p, 4) == b"\x7fELF" {
        log!("cubin (SASS) images cannot run on this device; PTX is required");
        return Err(CUDA_ERROR_NO_BINARY_FOR_GPU);
    }
    let text = std::ffi::CStr::from_ptr(p as *const std::ffi::c_char).to_string_lossy();
    load_ptx(dev, &text)
}

pub fn load_ptx(dev: usize, ptx: &str) -> Res<Box<Module>> {
    let d = drv()?;
    let device = &d.devices[dev];
    let opts = ptx2spirv::Options { float_atomics: device.float_atomics, fp64: device.fp64, warp_size: 32 };
    let out = ptx2spirv::compile_ptx(ptx, &opts).map_err(|e| {
        log!("PTX compilation failed: {e}");
        if let Ok(dir) = std::env::var("IGNUDA_DUMP") {
            let _ = std::fs::write(format!("{dir}/failed.ptx"), ptx);
        }
        CUDA_ERROR_INVALID_PTX
    })?;
    let spv = out.spirv_bytes();
    if let Ok(dir) = std::env::var("IGNUDA_DUMP") {
        let h = fxhash(ptx.as_bytes());
        let _ = std::fs::write(format!("{dir}/{h:016x}.ptx"), ptx);
        let _ = std::fs::write(format!("{dir}/{h:016x}.spv"), &spv);
    }
    let mut err = 0;
    let prog = unsafe { (d.cl.clCreateProgramWithIL)(device.cl_ctx, spv.as_ptr() as *const c_void, spv.len(), &mut err) };
    cl_check(err, "clCreateProgramWithIL").map_err(|_| CUDA_ERROR_INVALID_IMAGE)?;
    let build_opts = CString::new("-cl-fp32-correctly-rounded-divide-sqrt").unwrap();
    let r = unsafe { (d.cl.clBuildProgram)(prog, 1, &device.id, build_opts.as_ptr(), null(), null_mut()) };
    if r != 0 {
        let log = unsafe {
            let mut n = 0usize;
            (d.cl.clGetProgramBuildInfo)(prog, device.id, CL_PROGRAM_BUILD_LOG, 0, null_mut(), &mut n);
            let mut b = vec![0u8; n];
            (d.cl.clGetProgramBuildInfo)(prog, device.id, CL_PROGRAM_BUILD_LOG, n, b.as_mut_ptr() as *mut c_void, null_mut());
            String::from_utf8_lossy(&b).to_string()
        };
        log!("clBuildProgram failed ({r}):\n{log}");
        unsafe { (d.cl.clReleaseProgram)(prog) };
        return Err(CUDA_ERROR_INVALID_PTX);
    }
    let mut m = Box::new(Module { dev, program: prog, functions: HashMap::new(), globals: HashMap::new() });
    let mp = &*m as *const Module;
    for k in &out.kernels {
        let name = CString::new(k.name.clone()).unwrap();
        let kern = unsafe { (d.cl.clCreateKernel)(prog, name.as_ptr(), &mut err) };
        cl_check(err, "clCreateKernel")?;
        let t: cl_bool = 1;
        for info in [
            CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
        ] {
            unsafe { (d.cl.clSetKernelExecInfo)(kern, info, 4, &t as *const u32 as *const c_void) };
        }
        let wg_info = |p: u32| -> u64 {
            let mut v = 0u64;
            unsafe { (d.cl.clGetKernelWorkGroupInfo)(kern, device.id, p, 8, &mut v as *mut u64 as *mut c_void, null_mut()) };
            v
        };
        let f = Function {
            kernel: Mutex::new(kern),
            info: k.clone(),
            module: mp,
            max_threads: wg_info(CL_KERNEL_WORK_GROUP_SIZE) as usize,
            local_mem: wg_info(CL_KERNEL_LOCAL_MEM_SIZE),
            private_mem: wg_info(CL_KERNEL_PRIVATE_MEM_SIZE),
            attrs: Mutex::new(HashMap::new()),
        };
        m.functions.insert(k.name.clone(), Box::new(f));
    }
    if !out.globals.is_empty() {
        // Ask the device for the addresses of module globals.
        let n = out.globals.len();
        let q = current()?.queue;
        let buf = mem_alloc(8 * n, AllocKind::Host)?;
        let name = CString::new(ptx2spirv::GLOBALS_KERNEL).unwrap();
        let kern = unsafe { (d.cl.clCreateKernel)(prog, name.as_ptr(), &mut err) };
        cl_check(err, "clCreateKernel(globals)")?;
        unsafe {
            cl_check((device.usm.clSetKernelArgMemPointerINTEL)(kern, 0, buf as *const c_void), "set globals arg")?;
            let one = 1usize;
            cl_check((d.cl.clEnqueueNDRangeKernel)(q, kern, 1, null(), &one, &one, 0, null(), null_mut()), "enqueue globals")?;
            cl_check((d.cl.clFinish)(q), "clFinish")?;
            (d.cl.clReleaseKernel)(kern);
        }
        let addrs = unsafe { std::slice::from_raw_parts(buf as *const u64, n) }.to_vec();
        mem_free(buf)?;
        let mut gm = d.globals.lock().unwrap();
        for (g, a) in out.globals.iter().zip(addrs) {
            log!("global {} at {a:#x} ({} bytes)", g.name, g.size);
            m.globals.insert(g.name.clone(), (a, g.size));
            gm.insert(a, g.size);
        }
    }
    log!("loaded module: {} kernels, {} globals", m.functions.len(), m.globals.len());
    Ok(m)
}

pub fn unload(m: Box<Module>) -> Res {
    let d = drv()?;
    let mut gm = d.globals.lock().unwrap();
    for (a, _) in m.globals.values() {
        gm.remove(a);
    }
    for f in m.functions.values() {
        unsafe { (d.cl.clReleaseKernel)(*f.kernel.lock().unwrap()) };
    }
    unsafe { (d.cl.clReleaseProgram)(m.program) };
    Ok(())
}

fn fxhash(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// ------------------------------------------------------------------- launch

/// # Safety
/// `params`/`extra` follow cuLaunchKernel's contract.
pub unsafe fn launch(
    f: &Function,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    stream: CUstream,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Res {
    let d = drv()?;
    let q = queue_of(stream)?;
    let k = f.kernel.lock().unwrap();
    let info = &f.info;
    let threads = block[0] as usize * block[1] as usize * block[2] as usize;
    if threads == 0 || grid.iter().any(|&g| g == 0) {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    if threads > f.max_threads.max(1) {
        log!("launch of {} with {threads} threads exceeds the kernel limit {}", info.name, f.max_threads);
        return Err(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    }
    if !params.is_null() {
        for (i, p) in info.params.iter().enumerate() {
            let v = *params.add(i);
            cl_check((d.cl.clSetKernelArg)(*k, i as u32, p.size as usize, v), "clSetKernelArg")?;
        }
    } else if !extra.is_null() {
        // CU_LAUNCH_PARAM_BUFFER_POINTER, ptr, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END
        let mut buf: *const u8 = null();
        let mut i = 0;
        loop {
            let key = *extra.add(i) as usize;
            match key {
                0 => break, // END
                1 => buf = *extra.add(i + 1) as *const u8,
                2 => {}
                _ => return Err(CUDA_ERROR_INVALID_VALUE),
            }
            i += 2;
        }
        if buf.is_null() && !info.params.is_empty() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let mut off = 0usize;
        for (i, p) in info.params.iter().enumerate() {
            let a = p.align.max(1) as usize;
            off = (off + a - 1) / a * a;
            cl_check((d.cl.clSetKernelArg)(*k, i as u32, p.size as usize, buf.add(off) as *const c_void), "clSetKernelArg")?;
            off += p.size as usize;
        }
    } else if !info.params.is_empty() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    if let Some(idx) = info.dyn_shared_arg {
        cl_check((d.cl.clSetKernelArg)(*k, idx, shared.max(4) as usize, null()), "clSetKernelArg(shared)")?;
    }
    let local = [block[0] as usize, block[1] as usize, block[2] as usize];
    let global = [
        grid[0] as usize * local[0],
        grid[1] as usize * local[1],
        grid[2] as usize * local[2],
    ];
    log!("launch {} grid {:?} block {:?} shared {}", info.name, grid, block, shared);
    cl_check(
        (d.cl.clEnqueueNDRangeKernel)(q, *k, 3, null(), global.as_ptr(), local.as_ptr(), 0, null(), null_mut()),
        "clEnqueueNDRangeKernel",
    )?;
    if std::env::var_os("IGNUDA_SYNC").is_some() {
        cl_check((d.cl.clFinish)(q), "clFinish")?;
    }
    Ok(())
}
