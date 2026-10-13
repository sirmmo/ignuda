//! PTX abstract syntax tree.
//!
//! The parser is deliberately shallow: instructions keep their opcode and
//! dot-modifiers as strings, and the compiler interprets them. That keeps the
//! grammar small and makes unsupported instructions a compile-time error with
//! a useful message instead of a parse failure.

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub version: (u32, u32),
    pub target: Vec<String>,
    pub address_size: u32,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Var(VarDecl),
    Func(Function),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateSpace {
    Reg,
    Sreg,
    Const,
    Global,
    Local,
    Param,
    Shared,
    Tex,
    Generic,
}

impl StateSpace {
    pub fn from_dir(s: &str) -> Option<Self> {
        Some(match s {
            "reg" => Self::Reg,
            "sreg" => Self::Sreg,
            "const" => Self::Const,
            "global" => Self::Global,
            "local" => Self::Local,
            "param" => Self::Param,
            "shared" => Self::Shared,
            "tex" => Self::Tex,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linkage {
    #[default]
    None,
    Visible,
    Extern,
    Weak,
    Common,
}

/// Scalar PTX types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    B8,
    B16,
    B32,
    B64,
    B128,
    U8,
    U16,
    U32,
    U64,
    S8,
    S16,
    S32,
    S64,
    F16,
    F16x2,
    BF16,
    BF16x2,
    F32,
    F64,
    Pred,
    TexRef,
    SamplerRef,
    SurfRef,
}

impl Scalar {
    pub fn from_dir(s: &str) -> Option<Self> {
        Some(match s {
            "b8" => Self::B8,
            "b16" => Self::B16,
            "b32" => Self::B32,
            "b64" => Self::B64,
            "b128" => Self::B128,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "s8" => Self::S8,
            "s16" => Self::S16,
            "s32" => Self::S32,
            "s64" => Self::S64,
            "f16" => Self::F16,
            "f16x2" => Self::F16x2,
            "bf16" => Self::BF16,
            "bf16x2" => Self::BF16x2,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "pred" => Self::Pred,
            "texref" => Self::TexRef,
            "samplerref" => Self::SamplerRef,
            "surfref" => Self::SurfRef,
            _ => return None,
        })
    }

    /// Size in bytes (predicates count as 1).
    pub fn size(self) -> u32 {
        use Scalar::*;
        match self {
            B8 | U8 | S8 | Pred => 1,
            B16 | U16 | S16 | F16 | BF16 => 2,
            B32 | U32 | S32 | F32 | F16x2 | BF16x2 => 4,
            B64 | U64 | S64 | F64 | TexRef | SamplerRef | SurfRef => 8,
            B128 => 16,
        }
    }

    pub fn kind(self) -> ScalarKind {
        use Scalar::*;
        match self {
            S8 | S16 | S32 | S64 => ScalarKind::Signed,
            U8 | U16 | U32 | U64 => ScalarKind::Unsigned,
            B8 | B16 | B32 | B64 | B128 | F16x2 | BF16x2 | TexRef | SamplerRef | SurfRef => {
                ScalarKind::Bits
            }
            F16 | BF16 | F32 | F64 => ScalarKind::Float,
            Pred => ScalarKind::Pred,
        }
    }

    pub fn is_float(self) -> bool {
        self.kind() == ScalarKind::Float
    }
    pub fn is_signed(self) -> bool {
        self.kind() == ScalarKind::Signed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Signed,
    Unsigned,
    Bits,
    Float,
    Pred,
}

/// A declared type: scalar, optionally a short vector (.v2/.v4/.v8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Type {
    pub scalar: Scalar,
    pub vector: u8, // 0 = scalar
}

impl Type {
    pub fn scalar(s: Scalar) -> Self {
        Type { scalar: s, vector: 0 }
    }
    pub fn size(&self) -> u32 {
        self.scalar.size() * (self.vector.max(1) as u32)
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub space: StateSpace,
    pub linkage: Linkage,
    pub align: Option<u32>,
    pub ty: Type,
    pub name: String,
    /// Array dimensions; `None` means unsized (`[]`).
    pub dims: Vec<Option<u64>>,
    /// `%r<N>` parameterized register declaration count.
    pub count: Option<u32>,
    pub init: Option<Init>,
    pub line: u32,
}

impl VarDecl {
    /// Total element count (product of sized dimensions; unsized counts as 0).
    pub fn elements(&self) -> u64 {
        self.dims.iter().map(|d| d.unwrap_or(0)).product::<u64>()
    }
    pub fn byte_size(&self) -> u64 {
        self.ty.size() as u64 * if self.dims.is_empty() { 1 } else { self.elements() }
    }
}

#[derive(Debug, Clone)]
pub enum Init {
    Expr(Expr),
    List(Vec<Init>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i128),
    F32(u32),
    F64(u64),
    /// Address of a symbol, plus byte offset. `generic` is true for `generic(sym)`.
    Sym { name: String, offset: i64, generic: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuncKind {
    Entry,
    Func,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub kind: FuncKind,
    pub linkage: Linkage,
    pub name: String,
    pub ret: Vec<VarDecl>,
    pub params: Vec<VarDecl>,
    pub body: Option<Vec<Stmt>>,
    pub maxntid: Option<[u32; 3]>,
    pub reqntid: Option<[u32; 3]>,
    pub noreturn: bool,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Decl(VarDecl),
    Label(String),
    Inst(Inst),
    Block(Vec<Stmt>),
    /// `name: .callprototype (ret) _ (params);`
    CallProto { name: String, ret: Vec<VarDecl>, params: Vec<VarDecl> },
}

#[derive(Debug, Clone)]
pub struct Inst {
    /// Guard predicate: (negated, register name).
    pub pred: Option<(bool, String)>,
    pub op: String,
    pub mods: Vec<String>,
    pub args: Vec<Operand>,
    pub line: u32,
}

impl Inst {
    pub fn has(&self, m: &str) -> bool {
        self.mods.iter().any(|x| x == m)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Register, variable, label or function name; optional vector component
    /// (`%tid.x` -> comp = Some(0)).
    Id { name: String, comp: Option<u8> },
    /// `name+offset` outside of brackets.
    SymOff { name: String, offset: i64 },
    Int(i128),
    F32(u32),
    F64(u64),
    /// `[base+offset]`; base `None` means an absolute address.
    Addr { base: Option<String>, offset: i64 },
    /// `[a, b, ...]` (texture/surface operands).
    AddrList(Vec<Operand>),
    /// `{a, b, c, d}`
    Vector(Vec<Operand>),
    /// `(a, b)` call argument lists.
    List(Vec<Operand>),
    /// `!p`
    Not(Box<Operand>),
    /// `p|q` (setp with two destinations)
    Pair(Box<Operand>, Box<Operand>),
}

impl Operand {
    pub fn id(&self) -> Option<&str> {
        match self {
            Operand::Id { name, comp: None } => Some(name),
            _ => None,
        }
    }
}
