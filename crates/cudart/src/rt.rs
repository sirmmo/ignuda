//! Runtime API entry points and the nvcc registration glue.
#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc, clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;

use crate::drv::{self, CUcontext, CUevent, CUfunction, CUmodule, CUstream, Drv};

pub type cudaError_t = c_int;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorInvalidDevice: cudaError_t = 101;
pub const cudaErrorInvalidDeviceFunction: cudaError_t = 98;
pub const cudaErrorNoDevice: cudaError_t = 100;
pub const cudaErrorNotReady: cudaError_t = 600;
pub const cudaErrorUnknown: cudaError_t = 999;

/// Map a driver CUresult onto the matching runtime error (they share the low
/// numeric range for the common cases).
fn from_cu(e: drv::CUresult) -> cudaError_t {
    match e {
        0 => cudaSuccess,
        1 => cudaErrorInvalidValue,
        2 => cudaErrorMemoryAllocation,
        3 => cudaErrorInitializationError,
        100 => cudaErrorNoDevice,
        101 => cudaErrorInvalidDevice,
        209 | 218 => cudaErrorInvalidDeviceFunction,
        600 => cudaErrorNotReady,
        _ => cudaErrorUnknown,
    }
}

thread_local! {
    static LAST_ERROR: std::cell::Cell<cudaError_t> = const { std::cell::Cell::new(cudaSuccess) };
    /// Stack of launch configurations pushed by __cudaPushCallConfiguration.
    static CALL_CONFIG: RefCell<Vec<CallConfig>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct CallConfig {
    grid: Dim3,
    block: Dim3,
    shared: usize,
    stream: CUstream,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

fn set_err(e: cudaError_t) -> cudaError_t {
    LAST_ERROR.with(|c| c.set(e));
    e
}

fn d() -> Result<&'static Drv, cudaError_t> {
    drv::get().ok_or(cudaErrorInitializationError)
}

macro_rules! cu {
    ($call:expr) => {{
        let r = $call;
        if r != 0 {
            return set_err(from_cu(r));
        }
    }};
}

// ----------------------------------------------------------- module registry

struct Kernel {
    func: CUfunction,
    name: CString,
}

/// A registered fat binary and the host-stub -> kernel map built from it.
#[derive(Default)]
struct Registry {
    modules: Vec<Module>,
    /// host stub address -> (module index, device kernel name)
    functions: HashMap<usize, (usize, CString)>,
    /// host var address -> (module index, device symbol name)
    variables: HashMap<usize, (usize, CString)>,
    initialized: bool,
}

struct Module {
    image: *const c_void,
    cumodule: CUmodule,
    kernels: HashMap<String, Kernel>,
}

unsafe impl Send for Registry {}

fn registry() -> &'static Mutex<Registry> {
    static R: std::sync::OnceLock<Mutex<Registry>> = std::sync::OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::default()))
}

static DRIVER_READY: AtomicBool = AtomicBool::new(false);
static CURRENT_DEVICE: AtomicI32 = AtomicI32::new(0);

/// Lazily initialise the driver and bind the primary context of the current
/// device, exactly as the real runtime does on first use.
fn ensure_context() -> Result<(), cudaError_t> {
    let drv = d()?;
    if !DRIVER_READY.swap(true, Ordering::AcqRel) {
        unsafe {
            let r = (drv.cuInit)(0);
            if r != 0 {
                DRIVER_READY.store(false, Ordering::Release);
                return Err(from_cu(r));
            }
        }
    }
    let dev = CURRENT_DEVICE.load(Ordering::Relaxed);
    thread_local! {
        static BOUND: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    }
    BOUND.with(|b| -> Result<(), cudaError_t> {
        if b.get() == dev {
            return Ok(());
        }
        unsafe {
            let mut ctx: CUcontext = null_mut();
            let r = (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev);
            if r != 0 {
                return Err(from_cu(r));
            }
            (drv.cuCtxSetCurrent)(ctx);
        }
        b.set(dev);
        Ok(())
    })?;
    // Make sure every registered module is loaded into the current context.
    load_pending_modules(drv)
}

fn load_pending_modules(drv: &Drv) -> Result<(), cudaError_t> {
    let mut reg = registry().lock().unwrap();
    let count = reg.modules.len();
    for i in 0..count {
        if !reg.modules[i].cumodule.is_null() {
            continue;
        }
        let image = reg.modules[i].image;
        let mut m: CUmodule = null_mut();
        let r = unsafe { (drv.cuModuleLoadFatBinary)(&mut m, image) };
        if r != 0 {
            return Err(from_cu(r));
        }
        reg.modules[i].cumodule = m;
    }
    // Resolve any functions whose module is now loaded.
    let funcs: Vec<(usize, usize, CString)> =
        reg.functions.iter().map(|(&h, (mi, n))| (h, *mi, n.clone())).collect();
    for (host, mi, name) in funcs {
        if reg.modules[mi].kernels.contains_key(&host.to_string()) {
            continue;
        }
        let m = reg.modules[mi].cumodule;
        let mut f: CUfunction = null_mut();
        let r = unsafe { (drv.cuModuleGetFunction)(&mut f, m, name.as_ptr()) };
        if r == 0 {
            reg.modules[mi].kernels.insert(host.to_string(), Kernel { func: f, name });
        }
    }
    Ok(())
}

// ----------------------------------------------------- nvcc registration glue

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterFatBinary(fat: *mut c_void) -> *mut *mut c_void {
    // fat points at { int magic; int version; const void* data; void* filename }.
    let wrapper = fat as *const u32;
    let image = if !fat.is_null() && wrapper.read() == crate::drv_wrapper_magic() {
        *(fat.add(8) as *const *const c_void)
    } else {
        fat as *const c_void
    };
    let mut reg = registry().lock().unwrap();
    let idx = reg.modules.len();
    reg.modules.push(Module { image, cumodule: null_mut(), kernels: HashMap::new() });
    // The handle we return is an owned box holding the module index; nvcc only
    // ever passes it back to us, opaquely.
    Box::into_raw(Box::new(idx)) as *mut *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterFatBinaryEnd(_handle: *mut *mut c_void) {}

#[no_mangle]
pub unsafe extern "C" fn __cudaUnregisterFatBinary(handle: *mut *mut c_void) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle as *mut usize));
}

unsafe fn handle_index(handle: *mut *mut c_void) -> Option<usize> {
    if handle.is_null() {
        None
    } else {
        Some(*(handle as *const usize))
    }
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterFunction(
    handle: *mut *mut c_void,
    host_fun: *const c_char,
    _device_fun: *mut c_char,
    device_name: *const c_char,
    _thread_limit: c_int,
    _tid: *mut c_void,
    _bid: *mut c_void,
    _bdim: *mut c_void,
    _gdim: *mut c_void,
    _wsize: *mut c_int,
) {
    let Some(mi) = handle_index(handle) else { return };
    let name = CStr::from_ptr(device_name).to_owned();
    registry().lock().unwrap().functions.insert(host_fun as usize, (mi, name));
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterVar(
    handle: *mut *mut c_void,
    host_var: *mut c_char,
    _device_addr: *mut c_char,
    device_name: *const c_char,
    _ext: c_int,
    _size: usize,
    _constant: c_int,
    _global: c_int,
) {
    let Some(mi) = handle_index(handle) else { return };
    let name = CStr::from_ptr(device_name).to_owned();
    registry().lock().unwrap().variables.insert(host_var as usize, (mi, name));
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterManagedVar(
    handle: *mut *mut c_void,
    host_var: *mut *mut c_void,
    device_addr: *mut c_char,
    device_name: *const c_char,
    ext: c_int,
    size: usize,
    constant: c_int,
    global: c_int,
) {
    __cudaRegisterVar(handle, host_var as *mut c_char, device_addr, device_name, ext, size, constant, global);
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterTexture(
    _h: *mut *mut c_void,
    _hv: *const c_void,
    _da: *mut *mut c_void,
    _dn: *const c_char,
    _dim: c_int,
    _norm: c_int,
    _ext: c_int,
) {
}

#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterSurface(
    _h: *mut *mut c_void,
    _hv: *const c_void,
    _da: *mut *mut c_void,
    _dn: *const c_char,
    _dim: c_int,
    _ext: c_int,
) {
}

#[no_mangle]
pub unsafe extern "C" fn __cudaInitModule(_handle: *mut *mut c_void) -> c_char {
    1
}

#[no_mangle]
pub unsafe extern "C" fn __cudaPushCallConfiguration(
    grid: Dim3,
    block: Dim3,
    shared: usize,
    stream: CUstream,
) -> c_int {
    CALL_CONFIG.with(|c| c.borrow_mut().push(CallConfig { grid, block, shared, stream }));
    0
}

#[no_mangle]
pub unsafe extern "C" fn __cudaPopCallConfiguration(
    grid: *mut Dim3,
    block: *mut Dim3,
    shared: *mut usize,
    stream: *mut CUstream,
) -> c_int {
    match CALL_CONFIG.with(|c| c.borrow_mut().pop()) {
        Some(cfg) => {
            *grid = cfg.grid;
            *block = cfg.block;
            *shared = cfg.shared;
            *stream = cfg.stream;
            0
        }
        None => 1,
    }
}

// ----------------------------------------------------------------- launching

fn lookup_kernel(host: usize) -> Result<CUfunction, cudaError_t> {
    let reg = registry().lock().unwrap();
    let (mi, _) = reg.functions.get(&host).ok_or(cudaErrorInvalidDeviceFunction)?;
    reg.modules[*mi]
        .kernels
        .get(&host.to_string())
        .map(|k| k.func)
        .ok_or(cudaErrorInvalidDeviceFunction)
}

#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernel(
    func: *const c_void,
    grid: Dim3,
    block: Dim3,
    args: *mut *mut c_void,
    shared: usize,
    stream: CUstream,
) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    let f = match lookup_kernel(func as usize) {
        Ok(f) => f,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuLaunchKernel)(
        f, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared as c_uint, stream, args, null_mut()
    ));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernelExC(
    config: *const c_void,
    func: *const c_void,
    args: *mut *mut c_void,
) -> cudaError_t {
    // cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes;
    //                      cudaStream_t stream; cudaLaunchAttribute* attrs; uint numAttrs; }
    let p = config as *const u8;
    let grid = *(p as *const Dim3);
    let block = *(p.add(12) as *const Dim3);
    let shared = *(p.add(24) as *const usize);
    let stream = *(p.add(32) as *const CUstream);
    cudaLaunchKernel(func, grid, block, args, shared, stream)
}

// ------------------------------------------------------------------- devices

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t {
    if let Err(e) = ensure_context() {
        if !count.is_null() {
            *count = 0;
        }
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuDeviceGetCount)(count));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaSetDevice(dev: c_int) -> cudaError_t {
    CURRENT_DEVICE.store(dev, Ordering::Relaxed);
    set_err(match ensure_context() {
        Ok(()) => cudaSuccess,
        Err(e) => e,
    })
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetDevice(dev: *mut c_int) -> cudaError_t {
    if dev.is_null() {
        return set_err(cudaErrorInvalidValue);
    }
    *dev = CURRENT_DEVICE.load(Ordering::Relaxed);
    set_err(cudaSuccess)
}

/// CUDA 12 `cudaDeviceProp`, reproduced field-for-field in declaration order so
/// `#[repr(C)]` yields the exact ABI offsets a consumer (ggml) compiled against
/// the real header expects. The size assertion below pins it to CUDA 12.x's
/// `sizeof(cudaDeviceProp) == 1032`; the fields ggml reads (`name`, `major`,
/// `minor`, `multiProcessorCount`, `sharedMemPerBlock`, `sharedMemPerBlockOptin`,
/// `warpSize`, `integrated`) all sit in the stable middle of the struct.
#[repr(C)]
pub struct cudaDeviceProp {
    pub name: [c_char; 256],
    pub uuid: [c_char; 16],
    pub luid: [c_char; 8],
    pub luidDeviceNodeMask: c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseCudaArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_uint,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_uint,
    pub deferredMappingCudaArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved: [c_int; 63],
}

const _: () = assert!(std::mem::size_of::<cudaDeviceProp>() == 1032);

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, dev: c_int) -> cudaError_t {
    if prop.is_null() {
        return set_err(cudaErrorInvalidValue);
    }
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    std::ptr::write_bytes(prop as *mut u8, 0, std::mem::size_of::<cudaDeviceProp>());
    let p = &mut *prop;

    (drv.cuDeviceGetName)(p.name.as_mut_ptr(), p.name.len() as c_int, dev);
    (drv.cuDeviceGetUuid_v2)(p.uuid.as_mut_ptr() as *mut [u8; 16], dev);
    let mut tot: usize = 0;
    (drv.cuDeviceTotalMem_v2)(&mut tot, dev);
    p.totalGlobalMem = tot;

    // Driver CU_DEVICE_ATTRIBUTE_* values (the runtime enum differs; here we
    // query the driver directly, so these are the driver numbers).
    let a = |attr: c_int| -> c_int {
        let mut v = 0;
        (drv.cuDeviceGetAttribute)(&mut v, attr, dev);
        v
    };
    p.major = a(75);
    p.minor = a(76);
    p.warpSize = a(10);
    p.maxThreadsPerBlock = a(1);
    p.maxThreadsDim = [a(2), a(3), a(4)];
    p.maxGridSize = [a(5), a(6), a(7)];
    p.clockRate = a(13);
    p.totalConstMem = a(9) as usize;
    p.sharedMemPerBlock = a(8) as usize;
    p.sharedMemPerBlockOptin = a(97) as usize;
    p.sharedMemPerMultiprocessor = a(81) as usize;
    p.regsPerBlock = a(12);
    p.regsPerMultiprocessor = a(82);
    p.multiProcessorCount = a(16);
    p.maxThreadsPerMultiProcessor = a(39);
    p.maxBlocksPerMultiProcessor = a(106);
    p.integrated = a(18);
    p.canMapHostMemory = a(19);
    p.unifiedAddressing = a(41);
    p.managedMemory = a(83);
    p.concurrentKernels = a(31);
    p.asyncEngineCount = a(40);
    p.cooperativeLaunch = a(95);
    p.memoryClockRate = a(36);
    p.memoryBusWidth = a(37);
    p.l2CacheSize = a(38);
    p.textureAlignment = a(14) as usize;
    p.pciBusID = a(33);
    p.pciDeviceID = a(34);
    p.pciDomainID = a(50);
    set_err(cudaSuccess)
}

/// CUDA 12 `#define`s `cudaGetDeviceProperties` to the `_v2` symbol; a few
/// binaries still import the plain name. Both fill the CUDA 12 `cudaDeviceProp`.
#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, dev: c_int) -> cudaError_t {
    cudaGetDeviceProperties_v2(prop, dev)
}

#[no_mangle]
pub unsafe extern "C" fn cudaSetDeviceFlags(_flags: c_uint) -> cudaError_t {
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t {
    if !flags.is_null() {
        *flags = 0;
    }
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetPCIBusId(buf: *mut c_char, len: c_int, dev: c_int) -> cudaError_t {
    if buf.is_null() || len <= 0 {
        return set_err(cudaErrorInvalidValue);
    }
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuDeviceGetPCIBusId)(buf, len, dev));
    set_err(cudaSuccess)
}

/// 2D (pitched) copies. ignuda's memory is USM, so host and device pointers are
/// both plain addresses; copy row by row through the 1D driver copy.
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    _kind: c_int,
    stream: CUstream,
) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    for row in 0..height {
        let s = (src as u64) + (row * spitch) as u64;
        let dd = (dst as u64) + (row * dpitch) as u64;
        cu!((drv.cuMemcpyAsync)(dd, s, width, stream));
    }
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2D(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_int,
) -> cudaError_t {
    let r = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, null_mut());
    if r != cudaSuccess {
        return r;
    }
    cudaDeviceSynchronize()
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSynchronize() -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuCtxSynchronize)());
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceReset() -> cudaError_t {
    cudaDeviceSynchronize()
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetLastError() -> cudaError_t {
    LAST_ERROR.with(|c| {
        let e = c.get();
        c.set(cudaSuccess);
        e
    })
}

#[no_mangle]
pub unsafe extern "C" fn cudaPeekAtLastError() -> cudaError_t {
    LAST_ERROR.with(|c| c.get())
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetErrorString(e: cudaError_t) -> *const c_char {
    static STRINGS: Mutex<Option<HashMap<cudaError_t, CString>>> = Mutex::new(None);
    let text = match e {
        cudaSuccess => "no error",
        cudaErrorInvalidValue => "invalid argument",
        cudaErrorMemoryAllocation => "out of memory",
        cudaErrorInitializationError => "initialization error",
        cudaErrorNoDevice => "no CUDA-capable device is detected",
        cudaErrorInvalidDevice => "invalid device ordinal",
        cudaErrorInvalidDeviceFunction => "invalid device function",
        cudaErrorNotReady => "device not ready",
        _ => "unknown error",
    };
    let mut g = STRINGS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    m.entry(e).or_insert_with(|| CString::new(text).unwrap()).as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetErrorName(e: cudaError_t) -> *const c_char {
    cudaGetErrorString(e)
}

#[no_mangle]
pub unsafe extern "C" fn cudaDriverGetVersion(v: *mut c_int) -> cudaError_t {
    if v.is_null() {
        return set_err(cudaErrorInvalidValue);
    }
    match d() {
        Ok(drv) => {
            cu!((drv.cuDriverGetVersion)(v));
            set_err(cudaSuccess)
        }
        Err(e) => set_err(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn cudaRuntimeGetVersion(v: *mut c_int) -> cudaError_t {
    if v.is_null() {
        return set_err(cudaErrorInvalidValue);
    }
    *v = 12040;
    set_err(cudaSuccess)
}

// -------------------------------------------------------------------- memory

#[no_mangle]
pub unsafe extern "C" fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    let mut p: u64 = 0;
    cu!((drv.cuMemAlloc_v2)(&mut p, size));
    *ptr = p as *mut c_void;
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    let mut p: u64 = 0;
    cu!((drv.cuMemAllocManaged)(&mut p, size, if flags == 0 { 1 } else { flags }));
    *ptr = p as *mut c_void;
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaFree(ptr: *mut c_void) -> cudaError_t {
    if ptr.is_null() {
        return set_err(cudaSuccess);
    }
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuMemFree_v2)(ptr as u64));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    cudaHostAlloc(ptr, size, 0)
}

#[no_mangle]
pub unsafe extern "C" fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuMemHostAlloc)(ptr, size, flags));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuMemFreeHost)(ptr));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuMemHostRegister_v2)(ptr, size, flags));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuMemHostUnregister)(ptr));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: c_uint) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    let mut p: u64 = 0;
    cu!((drv.cuMemHostGetDevicePointer_v2)(&mut p, host, flags));
    *dev = p as *mut c_void;
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuMemGetInfo_v2)(free, total));
    set_err(cudaSuccess)
}

const MEMCPY_DEFAULT: c_int = 4;

#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, _kind: c_int) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    // With unified addressing the kind is advisory; the driver routes by pointer.
    cu!((drv.cuMemcpy)(dst as u64, src as u64, n));
    cu!((drv.cuCtxSynchronize)());
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    n: usize,
    _kind: c_int,
    stream: CUstream,
) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuMemcpyAsync)(dst as u64, src as u64, n, stream));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemset(dst: *mut c_void, value: c_int, n: usize) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuMemsetD8_v2)(dst as u64, value as u8, n));
    cu!((drv.cuCtxSynchronize)());
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemsetAsync(dst: *mut c_void, value: c_int, n: usize, stream: CUstream) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuMemsetD8Async)(dst as u64, value as u8, n, stream));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToSymbol(
    symbol: *const c_void,
    src: *const c_void,
    n: usize,
    offset: usize,
    _kind: c_int,
) -> cudaError_t {
    match symbol_address(symbol) {
        Ok(a) => cudaMemcpy((a + offset as u64) as *mut c_void, src, n, MEMCPY_DEFAULT),
        Err(e) => set_err(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromSymbol(
    dst: *mut c_void,
    symbol: *const c_void,
    n: usize,
    offset: usize,
    _kind: c_int,
) -> cudaError_t {
    match symbol_address(symbol) {
        Ok(a) => cudaMemcpy(dst, (a + offset as u64) as *const c_void, n, MEMCPY_DEFAULT),
        Err(e) => set_err(e),
    }
}

unsafe fn symbol_address(symbol: *const c_void) -> Result<u64, cudaError_t> {
    ensure_context()?;
    let drv = d()?;
    let reg = registry().lock().unwrap();
    let (mi, name) = reg.variables.get(&(symbol as usize)).ok_or(cudaErrorInvalidValue)?;
    let m = reg.modules[*mi].cumodule;
    let mut a: u64 = 0;
    let mut sz: usize = 0;
    let r = (drv.cuModuleGetGlobal_v2)(&mut a, &mut sz, m, name.as_ptr());
    if r != 0 {
        return Err(from_cu(r));
    }
    Ok(a)
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetSymbolAddress(ptr: *mut *mut c_void, symbol: *const c_void) -> cudaError_t {
    match symbol_address(symbol) {
        Ok(a) => {
            *ptr = a as *mut c_void;
            set_err(cudaSuccess)
        }
        Err(e) => set_err(e),
    }
}

// ------------------------------------------------------------------ streams

#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreate(stream: *mut CUstream) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuStreamCreate)(stream, 0));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreateWithFlags(stream: *mut CUstream, flags: c_uint) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuStreamCreate)(stream, flags));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreateWithPriority(stream: *mut CUstream, flags: c_uint, priority: c_int) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuStreamCreateWithPriority)(stream, flags, priority));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamDestroy(stream: CUstream) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuStreamDestroy_v2)(stream));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamSynchronize(stream: CUstream) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuStreamSynchronize)(stream));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamQuery(stream: CUstream) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    set_err(from_cu((drv.cuStreamQuery)(stream)))
}

#[no_mangle]
pub unsafe extern "C" fn cudaStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuStreamWaitEvent)(stream, event, flags));
    set_err(cudaSuccess)
}

// ------------------------------------------------------------------- events

#[no_mangle]
pub unsafe extern "C" fn cudaEventCreate(event: *mut CUevent) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuEventCreate)(event, 0));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventCreateWithFlags(event: *mut CUevent, flags: c_uint) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    cu!((drv.cuEventCreate)(event, flags));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventDestroy(event: CUevent) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuEventDestroy_v2)(event));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventRecord(event: CUevent, stream: CUstream) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuEventRecord)(event, stream));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventSynchronize(event: CUevent) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuEventSynchronize)(event));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventQuery(event: CUevent) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    set_err(from_cu((drv.cuEventQuery)(event)))
}

#[no_mangle]
pub unsafe extern "C" fn cudaEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> cudaError_t {
    let drv = match d() {
        Ok(x) => x,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuEventElapsedTime)(ms, start, end));
    set_err(cudaSuccess)
}

// -------------------------------------------------------------- attributes

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, dev: c_int) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    // Runtime and driver attribute enums differ; map the ones that do.
    let cu_attr = crate::map_device_attr(attr);
    if cu_attr < 0 {
        *value = 0;
        return set_err(cudaSuccess);
    }
    cu!((drv.cuDeviceGetAttribute)(value, cu_attr, dev));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetLimit(_limit: c_int, _value: usize) -> cudaError_t {
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaFuncGetAttributes(attr: *mut c_void, func: *const c_void) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    let f = match lookup_kernel(func as usize) {
        Ok(f) => f,
        Err(e) => return set_err(e),
    };
    // struct cudaFuncAttributes { size_t sharedSizeBytes; size_t constSizeBytes;
    //   size_t localSizeBytes; int maxThreadsPerBlock; int numRegs; int ptxVersion;
    //   int binaryVersion; int cacheModeCA; int maxDynamicSharedSizeBytes;
    //   int preferredShmemCarveout; int clusterDimMustBeSet; ... }
    if attr.is_null() {
        return set_err(cudaErrorInvalidValue);
    }
    std::ptr::write_bytes(attr as *mut u8, 0, 72);
    let get = |a: c_int| -> c_int {
        let mut v = 0;
        (drv.cuFuncGetAttribute)(&mut v, a, f);
        v
    };
    let p = attr as *mut u8;
    *(p as *mut usize) = get(1) as usize; // sharedSizeBytes (CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=1)
    *(p.add(8) as *mut usize) = get(2) as usize; // constSizeBytes
    *(p.add(16) as *mut usize) = get(3) as usize; // localSizeBytes
    *(p.add(24) as *mut c_int) = get(0); // maxThreadsPerBlock
    *(p.add(28) as *mut c_int) = get(4); // numRegs
    *(p.add(32) as *mut c_int) = get(5); // ptxVersion
    *(p.add(36) as *mut c_int) = get(6); // binaryVersion
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    blocks: *mut c_int,
    func: *const c_void,
    block_size: c_int,
    dynamic_smem: usize,
) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    let f = match lookup_kernel(func as usize) {
        Ok(f) => f,
        Err(e) => return set_err(e),
    };
    cu!((drv.cuOccupancyMaxActiveBlocksPerMultiprocessor)(blocks, f, block_size, dynamic_smem));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    blocks: *mut c_int,
    func: *const c_void,
    block_size: c_int,
    dynamic_smem: usize,
    _flags: c_uint,
) -> cudaError_t {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, func, block_size, dynamic_smem)
}

#[no_mangle]
pub unsafe extern "C" fn cudaThreadSynchronize() -> cudaError_t {
    cudaDeviceSynchronize()
}

#[no_mangle]
pub unsafe extern "C" fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> cudaError_t {
    if let Err(e) = ensure_context() {
        return set_err(e);
    }
    let drv = d().unwrap();
    let f = match lookup_kernel(func as usize) {
        Ok(f) => f,
        Err(e) => return set_err(e),
    };
    // Runtime attr 8 (maxDynamicSharedMemorySize) -> driver attr 8.
    cu!((drv.cuFuncSetAttribute)(f, attr, value));
    set_err(cudaSuccess)
}

#[no_mangle]
pub unsafe extern "C" fn cudaFuncSetCacheConfig(_func: *const c_void, _config: c_int) -> cudaError_t {
    set_err(cudaSuccess)
}
