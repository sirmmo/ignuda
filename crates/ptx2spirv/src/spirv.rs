//! Minimal SPIR-V module builder for the OpenCL (Kernel) environment.
//!
//! Types and constants are de-duplicated; sections are kept separate and
//! concatenated in the order the spec requires when `finish()` is called.

use std::collections::{BTreeSet, HashMap};

pub type Id = u32;

pub mod op {
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const COPY_MEMORY_SIZED: u16 = 64;
    pub const IN_BOUNDS_PTR_ACCESS_CHAIN: u16 = 70;
    pub const PTR_ACCESS_CHAIN: u16 = 67;
    pub const DECORATE: u16 = 71;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const COMPOSITE_INSERT: u16 = 82;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const F_CONVERT: u16 = 115;
    pub const CONVERT_PTR_TO_U: u16 = 117;
    pub const CONVERT_U_TO_PTR: u16 = 120;
    pub const PTR_CAST_TO_GENERIC: u16 = 121;
    pub const GENERIC_CAST_TO_PTR: u16 = 122;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const F_REM: u16 = 140;
    pub const I_ADD_CARRY: u16 = 149;
    pub const I_SUB_BORROW: u16 = 150;
    pub const U_MUL_EXTENDED: u16 = 151;
    pub const S_MUL_EXTENDED: u16 = 152;
    pub const IS_NAN: u16 = 156;
    pub const IS_INF: u16 = 157;
    pub const IS_FINITE: u16 = 158;
    pub const IS_NORMAL: u16 = 159;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const ORDERED: u16 = 162;
    pub const UNORDERED: u16 = 163;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const MEMORY_BARRIER: u16 = 225;
    pub const ATOMIC_LOAD: u16 = 227;
    pub const ATOMIC_STORE: u16 = 228;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_COMPARE_EXCHANGE: u16 = 230;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_I_SUB: u16 = 235;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const PHI: u16 = 245;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const RETURN: u16 = 253;
    pub const RETURN_VALUE: u16 = 254;
    pub const UNREACHABLE: u16 = 255;
    pub const GROUP_NON_UNIFORM_ALL: u16 = 334;
    pub const GROUP_NON_UNIFORM_ANY: u16 = 335;
    pub const GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
    pub const GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
    pub const GROUP_NON_UNIFORM_SHUFFLE_UP: u16 = 347;
    pub const GROUP_NON_UNIFORM_SHUFFLE_DOWN: u16 = 348;
    pub const ATOMIC_F_ADD_EXT: u16 = 6035;
}

pub mod cap {
    pub const ADDRESSES: u32 = 4;
    pub const LINKAGE: u32 = 5;
    pub const KERNEL: u32 = 6;
    pub const VECTOR16: u32 = 7;
    pub const FLOAT16_BUFFER: u32 = 8;
    pub const FLOAT16: u32 = 9;
    pub const FLOAT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const INT64_ATOMICS: u32 = 12;
    pub const INT16: u32 = 22;
    pub const GENERIC_POINTER: u32 = 38;
    pub const INT8: u32 = 39;
    pub const SUBGROUP_DISPATCH: u32 = 58;
    pub const GROUP_NON_UNIFORM: u32 = 61;
    pub const GROUP_NON_UNIFORM_VOTE: u32 = 62;
    pub const GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    pub const GROUP_NON_UNIFORM_SHUFFLE_RELATIVE: u32 = 66;
    pub const ATOMIC_FLOAT32_ADD_EXT: u32 = 6033;
    pub const ATOMIC_FLOAT64_ADD_EXT: u32 = 6034;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Storage {
    UniformConstant = 0,
    Input = 1,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    Generic = 8,
}

pub mod deco {
    pub const BUILTIN: u32 = 11;
    pub const VOLATILE: u32 = 21;
    pub const CONSTANT: u32 = 22;
    pub const SATURATED_CONVERSION: u32 = 28;
    pub const FUNC_PARAM_ATTR: u32 = 38;
    pub const FP_ROUNDING_MODE: u32 = 39;
    pub const LINKAGE_ATTRIBUTES: u32 = 41;
    pub const ALIGNMENT: u32 = 44;
}

pub mod builtin {
    pub const NUM_WORKGROUPS: u32 = 24;
    pub const WORKGROUP_SIZE: u32 = 25;
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const ENQUEUED_WORKGROUP_SIZE: u32 = 32;
    pub const SUBGROUP_SIZE: u32 = 36;
    pub const SUBGROUP_ID: u32 = 40;
    pub const SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
}

/// OpenCL.std extended instruction numbers.
pub mod cl {
    pub const CEIL: u32 = 12;
    pub const COPYSIGN: u32 = 13;
    pub const COS: u32 = 14;
    pub const EXP2: u32 = 20;
    pub const FABS: u32 = 23;
    pub const FLOOR: u32 = 25;
    pub const FMA: u32 = 26;
    pub const FMAX: u32 = 27;
    pub const FMIN: u32 = 28;
    pub const LOG2: u32 = 38;
    pub const RINT: u32 = 53;
    pub const ROUND: u32 = 55;
    pub const RSQRT: u32 = 56;
    pub const SIN: u32 = 57;
    pub const SQRT: u32 = 61;
    pub const TANH: u32 = 63;
    pub const TRUNC: u32 = 66;
    pub const NATIVE_COS: u32 = 81;
    pub const NATIVE_DIVIDE: u32 = 82;
    pub const NATIVE_EXP2: u32 = 84;
    pub const NATIVE_LOG2: u32 = 87;
    pub const NATIVE_RECIP: u32 = 90;
    pub const NATIVE_RSQRT: u32 = 91;
    pub const NATIVE_SIN: u32 = 92;
    pub const NATIVE_SQRT: u32 = 93;
    pub const S_ABS: u32 = 141;
    pub const S_ADD_SAT: u32 = 143;
    pub const U_ADD_SAT: u32 = 144;
    pub const CLZ: u32 = 151;
    pub const CTZ: u32 = 152;
    pub const S_MAX: u32 = 156;
    pub const U_MAX: u32 = 157;
    pub const S_MIN: u32 = 158;
    pub const U_MIN: u32 = 159;
    pub const S_MUL_HI: u32 = 160;
    pub const ROTATE: u32 = 161;
    pub const S_SUB_SAT: u32 = 162;
    pub const U_SUB_SAT: u32 = 163;
    pub const POPCOUNT: u32 = 166;
    pub const S_MUL24: u32 = 169;
    pub const U_MUL24: u32 = 170;
    pub const U_ABS: u32 = 201;
    pub const U_MUL_HI: u32 = 203;
    pub const PRINTF: u32 = 184;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKey {
    Void,
    Bool,
    Int(u32),
    Float(u32),
    Vector(Id, u32),
    Array(Id, Id),
    Struct(Vec<Id>),
    Pointer(Storage, Id),
    Function(Id, Vec<Id>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ConstKey {
    Scalar(Id, Vec<u32>),
    Bool(bool),
    Null(Id),
    Composite(Id, Vec<Id>),
}

#[derive(Default)]
pub struct Builder {
    next_id: Id,
    pub capabilities: BTreeSet<u32>,
    extensions: BTreeSet<String>,
    ext_imports: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    debug: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>, // types, constants, module-scope variables
    pub functions: Vec<u32>,
    /// Function-scope OpVariables of the function being built; they must be
    /// placed at the start of its entry block.
    pub local_vars: Vec<u32>,
    types: HashMap<TypeKey, Id>,
    type_info: HashMap<Id, TypeKey>,
    consts: HashMap<ConstKey, Id>,
    opencl_std: Option<Id>,
}

fn encode_str(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | opcode as u32);
    out.extend_from_slice(operands);
}

impl Builder {
    pub fn new() -> Self {
        let mut b = Builder { next_id: 1, ..Default::default() };
        for c in [cap::ADDRESSES, cap::KERNEL, cap::INT64, cap::INT8, cap::INT16, cap::GENERIC_POINTER] {
            b.capabilities.insert(c);
        }
        b
    }

    pub fn id(&mut self) -> Id {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    pub fn capability(&mut self, c: u32) {
        self.capabilities.insert(c);
    }
    pub fn extension(&mut self, e: &str) {
        self.extensions.insert(e.to_string());
    }

    pub fn opencl_std(&mut self) -> Id {
        if let Some(i) = self.opencl_std {
            return i;
        }
        let i = self.id();
        let mut ops = vec![i];
        ops.extend(encode_str("OpenCL.std"));
        inst(&mut self.ext_imports, op::EXT_INST_IMPORT, &ops);
        self.opencl_std = Some(i);
        i
    }

    // ---- types ----

    pub fn ty(&mut self, key: TypeKey) -> Id {
        if let Some(&i) = self.types.get(&key) {
            return i;
        }
        let i = self.id();
        match &key {
            TypeKey::Void => inst(&mut self.globals, op::TYPE_VOID, &[i]),
            TypeKey::Bool => inst(&mut self.globals, op::TYPE_BOOL, &[i]),
            TypeKey::Int(w) => {
                match w {
                    8 => self.capability(cap::INT8),
                    16 => self.capability(cap::INT16),
                    64 => self.capability(cap::INT64),
                    _ => {}
                }
                inst(&mut self.globals, op::TYPE_INT, &[i, *w, 0])
            }
            TypeKey::Float(w) => {
                match w {
                    16 => self.capability(cap::FLOAT16),
                    64 => self.capability(cap::FLOAT64),
                    _ => {}
                }
                inst(&mut self.globals, op::TYPE_FLOAT, &[i, *w])
            }
            TypeKey::Vector(e, n) => {
                if *n > 4 {
                    self.capability(cap::VECTOR16);
                }
                inst(&mut self.globals, op::TYPE_VECTOR, &[i, *e, *n])
            }
            TypeKey::Array(e, len) => inst(&mut self.globals, op::TYPE_ARRAY, &[i, *e, *len]),
            TypeKey::Struct(ms) => {
                let mut ops = vec![i];
                ops.extend(ms);
                inst(&mut self.globals, op::TYPE_STRUCT, &ops)
            }
            TypeKey::Pointer(s, t) => inst(&mut self.globals, op::TYPE_POINTER, &[i, *s as u32, *t]),
            TypeKey::Function(r, ps) => {
                let mut ops = vec![i, *r];
                ops.extend(ps);
                inst(&mut self.globals, op::TYPE_FUNCTION, &ops)
            }
        }
        self.types.insert(key.clone(), i);
        self.type_info.insert(i, key);
        i
    }

    pub fn type_of(&self, t: Id) -> &TypeKey {
        &self.type_info[&t]
    }

    pub fn void(&mut self) -> Id {
        self.ty(TypeKey::Void)
    }
    pub fn bool(&mut self) -> Id {
        self.ty(TypeKey::Bool)
    }
    pub fn int(&mut self, w: u32) -> Id {
        self.ty(TypeKey::Int(w))
    }
    pub fn float(&mut self, w: u32) -> Id {
        self.ty(TypeKey::Float(w))
    }
    pub fn vector(&mut self, e: Id, n: u32) -> Id {
        self.ty(TypeKey::Vector(e, n))
    }
    pub fn ptr(&mut self, s: Storage, t: Id) -> Id {
        self.ty(TypeKey::Pointer(s, t))
    }
    pub fn array(&mut self, e: Id, len: u32) -> Id {
        let l = self.const_u32(len);
        self.ty(TypeKey::Array(e, l))
    }
    pub fn fn_type(&mut self, ret: Id, params: Vec<Id>) -> Id {
        self.ty(TypeKey::Function(ret, params))
    }

    /// Bit width of a scalar int/float type, or 1 for bool.
    pub fn width(&self, t: Id) -> u32 {
        match self.type_of(t) {
            TypeKey::Int(w) | TypeKey::Float(w) => *w,
            TypeKey::Bool => 1,
            TypeKey::Vector(e, n) => self.width(*e) * n,
            TypeKey::Pointer(..) => 64,
            k => panic!("width of non-scalar type {k:?}"),
        }
    }

    // ---- constants ----

    fn constant(&mut self, key: ConstKey) -> Id {
        if let Some(&i) = self.consts.get(&key) {
            return i;
        }
        let i = self.id();
        match &key {
            ConstKey::Scalar(t, words) => {
                let mut ops = vec![*t, i];
                ops.extend(words);
                inst(&mut self.globals, op::CONSTANT, &ops)
            }
            ConstKey::Bool(true) => {
                let b = self.bool();
                inst(&mut self.globals, op::CONSTANT_TRUE, &[b, i])
            }
            ConstKey::Bool(false) => {
                let b = self.bool();
                inst(&mut self.globals, op::CONSTANT_FALSE, &[b, i])
            }
            ConstKey::Null(t) => inst(&mut self.globals, op::CONSTANT_NULL, &[*t, i]),
            ConstKey::Composite(t, parts) => {
                let mut ops = vec![*t, i];
                ops.extend(parts);
                inst(&mut self.globals, op::CONSTANT_COMPOSITE, &ops)
            }
        }
        self.consts.insert(key, i);
        i
    }

    /// Constant of scalar type `t` from raw bits (truncated to the type width).
    pub fn const_bits(&mut self, t: Id, bits: u64) -> Id {
        match self.type_of(t).clone() {
            TypeKey::Bool => self.constant(ConstKey::Bool(bits != 0)),
            TypeKey::Int(w) | TypeKey::Float(w) => {
                let words = match w {
                    8 => vec![(bits & 0xff) as u32],
                    16 => vec![(bits & 0xffff) as u32],
                    32 => vec![bits as u32],
                    _ => vec![bits as u32, (bits >> 32) as u32],
                };
                self.constant(ConstKey::Scalar(t, words))
            }
            k => panic!("const_bits on {k:?}"),
        }
    }
    pub fn const_u32(&mut self, v: u32) -> Id {
        let t = self.int(32);
        self.const_bits(t, v as u64)
    }
    pub fn const_u64(&mut self, v: u64) -> Id {
        let t = self.int(64);
        self.const_bits(t, v)
    }
    pub fn const_bool(&mut self, v: bool) -> Id {
        self.constant(ConstKey::Bool(v))
    }
    pub fn const_null(&mut self, t: Id) -> Id {
        self.constant(ConstKey::Null(t))
    }
    pub fn const_composite(&mut self, t: Id, parts: Vec<Id>) -> Id {
        self.constant(ConstKey::Composite(t, parts))
    }

    // ---- module-level declarations ----

    pub fn name(&mut self, target: Id, name: &str) {
        let mut ops = vec![target];
        ops.extend(encode_str(name));
        inst(&mut self.debug, op::NAME, &ops);
    }

    pub fn decorate(&mut self, target: Id, deco: u32, extra: &[u32]) {
        let mut ops = vec![target, deco];
        ops.extend(extra);
        inst(&mut self.annotations, op::DECORATE, &ops);
    }

    pub fn linkage(&mut self, target: Id, name: &str, export: bool) {
        self.capability(cap::LINKAGE);
        let mut ops = vec![target, deco::LINKAGE_ATTRIBUTES];
        ops.extend(encode_str(name));
        ops.push(if export { 0 } else { 1 });
        inst(&mut self.annotations, op::DECORATE, &ops);
    }

    pub fn entry_point(&mut self, func: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![6 /* Kernel */, func];
        ops.extend(encode_str(name));
        ops.extend(interface);
        inst(&mut self.entry_points, op::ENTRY_POINT, &ops);
    }

    pub fn execution_mode(&mut self, func: Id, mode: u32, args: &[u32]) {
        let mut ops = vec![func, mode];
        ops.extend(args);
        inst(&mut self.exec_modes, op::EXECUTION_MODE, &ops);
    }

    /// Module-scope variable.
    pub fn global_var(&mut self, ptr_ty: Id, storage: Storage, init: Option<Id>) -> Id {
        let i = self.id();
        let mut ops = vec![ptr_ty, i, storage as u32];
        ops.extend(init);
        inst(&mut self.globals, op::VARIABLE, &ops);
        i
    }

    // ---- function bodies ----

    /// Append an instruction to the function section.
    pub fn emit(&mut self, opcode: u16, operands: &[u32]) {
        inst(&mut self.functions, opcode, operands);
    }

    /// Append an instruction with a result: `opcode result_type result operands...`.
    pub fn op(&mut self, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let r = self.id();
        let mut ops = Vec::with_capacity(operands.len() + 2);
        ops.push(ty);
        ops.push(r);
        ops.extend_from_slice(operands);
        inst(&mut self.functions, opcode, &ops);
        r
    }

    /// Function-storage variable (collected into `local_vars`).
    pub fn local_var(&mut self, ptr_ty: Id) -> Id {
        let i = self.id();
        inst(&mut self.local_vars, op::VARIABLE, &[ptr_ty, i, Storage::Function as u32]);
        i
    }

    pub fn ext(&mut self, ty: Id, inst_no: u32, args: &[Id]) -> Id {
        let set = self.opencl_std();
        let mut ops = vec![set, inst_no];
        ops.extend_from_slice(args);
        self.op(op::EXT_INST, ty, &ops)
    }

    pub fn finish(self) -> Vec<u32> {
        let mut out = vec![0x0723_0203, 0x0001_0300 /* 1.3 */, 0x1957_0001 /* generator */, self.next_id, 0];
        for c in &self.capabilities {
            inst(&mut out, op::CAPABILITY, &[*c]);
        }
        for e in &self.extensions {
            inst(&mut out, op::EXTENSION, &encode_str(e));
        }
        out.extend(&self.ext_imports);
        inst(&mut out, op::MEMORY_MODEL, &[2 /* Physical64 */, 2 /* OpenCL */]);
        out.extend(&self.entry_points);
        out.extend(&self.exec_modes);
        out.extend(&self.debug);
        out.extend(&self.annotations);
        out.extend(&self.globals);
        out.extend(&self.functions);
        out
    }
}
