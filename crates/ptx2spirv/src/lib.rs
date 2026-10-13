//! PTX -> SPIR-V compiler targeting the OpenCL kernel environment
//! (Intel iGPUs via Level Zero / OpenCL, or any OpenCL 3.0 runtime with USM).
//!
//! Translation strategy:
//! * every PTX register becomes a Function-storage variable (the driver's
//!   mem2reg/SROA turns them back into SSA values);
//! * device pointers are plain 64-bit integers, converted to typed pointers at
//!   each memory access (`OpConvertUToPtr`), mirroring PTX's untyped model;
//! * control flow is unstructured (labels -> blocks), which the Kernel
//!   execution model permits;
//! * kernel parameters are copied into Function-storage variables so that
//!   `ld.param [name+off]` and taking a parameter's address work uniformly.

pub mod spirv;
mod translate;

pub use translate::compile;

#[derive(Clone, Debug)]
pub struct Options {
    /// Device supports cl_ext_float_atomics / SPV_EXT_shader_atomic_float_add.
    pub float_atomics: bool,
    /// Device has native fp64.
    pub fp64: bool,
    /// Required sub-group size for kernels that use warp-level operations.
    pub warp_size: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options { float_atomics: false, fp64: true, warp_size: 32 }
    }
}

#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: String,
    pub size: u32,
    pub align: u32,
}

#[derive(Clone, Debug)]
pub struct KernelInfo {
    pub name: String,
    pub params: Vec<ParamInfo>,
    /// Index of the hidden `__local` argument carrying dynamic shared memory.
    pub dyn_shared_arg: Option<u32>,
    pub reqntid: Option<[u32; 3]>,
    pub maxntid: Option<[u32; 3]>,
    /// Bytes of statically declared shared memory.
    pub static_shared: u64,
    pub uses_warp_ops: bool,
}

#[derive(Clone, Debug)]
pub struct GlobalInfo {
    pub name: String,
    pub size: u64,
    pub is_const: bool,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub spirv: Vec<u32>,
    pub kernels: Vec<KernelInfo>,
    /// Module-scope device variables. Their addresses are written, in this
    /// order, by the synthesized kernel [`GLOBALS_KERNEL`].
    pub globals: Vec<GlobalInfo>,
}

/// Name of the synthesized kernel `void k(ulong *out)` that stores the device
/// address of every module global (see [`Output::globals`]) into `out`.
pub const GLOBALS_KERNEL: &str = "__ignuda_global_addresses";

impl Output {
    pub fn spirv_bytes(&self) -> Vec<u8> {
        self.spirv.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// Parse and compile PTX source text.
pub fn compile_ptx(src: &str, opts: &Options) -> Result<Output, String> {
    let m = ptx::parse(src)?;
    compile(&m, opts)
}
