use std::collections::HashMap;

use ptx::{FuncKind, Function, Inst, Item, Linkage, Module, Operand, Scalar, ScalarKind, StateSpace, Stmt, VarDecl};

use crate::spirv::{builtin, cap, cl, deco, op, Builder, Id, Storage, TypeKey};
use crate::{GlobalInfo, KernelInfo, Options, Output, ParamInfo, GLOBALS_KERNEL};

type R<T> = Result<T, String>;

/// A memory-resident symbol (module global, `.local`/`.param`/`.shared` variable, parameter copy).
#[derive(Clone, Debug)]
struct Sym {
    /// Pointer to the variable (type `ptr(storage, ty)`).
    ptr: Id,
    storage: Storage,
    /// Dynamic shared memory: the pointer is the function's hidden argument.
    dyn_shared: bool,
}

#[derive(Clone, Debug)]
enum Var {
    Reg { var: Id, ty: Id, scalar: Scalar, vector: u8 },
    Mem(Sym),
}

struct FuncSig {
    id: Id,
    ret_ty: Id,
    /// Return values (each a `.param` or `.reg` declaration) and their SPIR-V types.
    rets: Vec<(VarDecl, Id)>,
    params: Vec<(VarDecl, Id)>,
}

#[derive(Default)]
struct FnState {
    scopes: Vec<HashMap<String, Var>>,
    labels: HashMap<String, Id>,
    protos: HashMap<String, ()>,
    cur_block: Id,
    terminated: bool,
    is_entry: bool,
    /// Return-value variables of a `.func`.
    rets: Vec<(Var, Id)>,
    ret_ty: Id,
    /// Hidden pointer to dynamic shared memory (Workgroup i8*).
    dyn_smem: Option<Id>,
    /// Carry flag for `.cc` arithmetic (u32 variable holding 0/1).
    carry: Option<Id>,
    uses_warp_ops: bool,
}

struct Compiler<'a> {
    b: Builder,
    opts: &'a Options,
    globals: HashMap<String, Sym>,
    global_list: Vec<(GlobalInfo, Id)>,
    funcs: HashMap<String, FuncSig>,
    builtins: HashMap<u32, Id>,
    has_dyn_shared: bool,
    static_shared: u64,
    entry_list: Vec<(Id, String)>,
    /// Workgroup variable that integer shared-memory addresses are relative to.
    shared_anchor: Option<Id>,
    f: FnState,
}

pub fn compile(m: &Module, opts: &Options) -> R<Output> {
    if m.address_size != 64 {
        return Err("only .address_size 64 is supported".into());
    }
    let mut c = Compiler {
        b: Builder::new(),
        opts,
        globals: HashMap::new(),
        global_list: vec![],
        funcs: HashMap::new(),
        builtins: HashMap::new(),
        has_dyn_shared: false,
        static_shared: 0,
        entry_list: vec![],
        shared_anchor: None,
        f: FnState::default(),
    };
    // Pass 1: module variables and function signatures.
    for item in &m.items {
        match item {
            Item::Var(v) => c.declare_global(v)?,
            Item::Func(f) => c.declare_func(f)?,
        }
    }
    // Pass 2: function bodies.
    let mut kernels = Vec::new();
    for item in &m.items {
        if let Item::Func(f) = item {
            if f.body.is_some() {
                let k = c.function(f).map_err(|e| format!("in function {}: {e}", f.name))?;
                kernels.extend(k);
            }
        }
    }
    c.globals_kernel();
    let builtin_ids: Vec<Id> = c.builtins.values().copied().collect();
    let entries: Vec<(Id, String)> = c.entry_list.drain(..).collect();
    for (id, name) in entries {
        c.b.entry_point(id, &name, &builtin_ids);
    }
    let globals = c.global_list.iter().map(|(g, _)| g.clone()).collect();
    Ok(Output { spirv: c.b.finish(), kernels, globals })
}

fn storage_of(space: StateSpace) -> Storage {
    match space {
        StateSpace::Global | StateSpace::Const => Storage::CrossWorkgroup,
        StateSpace::Shared => Storage::Workgroup,
        StateSpace::Local | StateSpace::Param | StateSpace::Reg => Storage::Function,
        _ => Storage::Generic,
    }
}

/// Memory-scope for the `.cta/.gpu/.sys` qualifier.
fn scope_of(i: &Inst) -> u32 {
    if i.has("cta") {
        2
    } else if i.has("sys") {
        0
    } else {
        1
    }
}

/// Last modifier that names a scalar type.
fn inst_type(i: &Inst) -> R<Scalar> {
    i.mods.iter().rev().find_map(|m| Scalar::from_dir(m)).ok_or_else(|| format!("line {}: {} has no type", i.line, i.op))
}

/// The last two type modifiers (`cvt.f32.s32` -> (f32, s32)).
fn two_types(i: &Inst) -> R<(Scalar, Scalar)> {
    let ts: Vec<Scalar> = i.mods.iter().filter_map(|m| Scalar::from_dir(m)).collect();
    if ts.len() < 2 {
        return Err(format!("line {}: {} needs two types", i.line, i.op));
    }
    Ok((ts[ts.len() - 2], ts[ts.len() - 1]))
}

/// Packed-half types (`.f16x2` / `.bf16x2`): two 16-bit floats in a 32-bit
/// register. ggml's `half2`/`__hadd2`/`__hfma2` paths lower to these.
fn is_packed_half(s: Scalar) -> bool {
    matches!(s, Scalar::F16x2 | Scalar::BF16x2)
}

fn space_of(i: &Inst) -> StateSpace {
    for m in &i.mods {
        match m.as_str() {
            "global" => return StateSpace::Global,
            "shared" => return StateSpace::Shared,
            "local" => return StateSpace::Local,
            "param" => return StateSpace::Param,
            "const" => return StateSpace::Const,
            _ => {}
        }
    }
    StateSpace::Generic
}

fn vec_width(i: &Inst) -> u8 {
    for m in &i.mods {
        match m.as_str() {
            "v2" => return 2,
            "v4" => return 4,
            "v8" => return 8,
            _ => {}
        }
    }
    0
}

fn arg<'i>(i: &'i Inst, n: usize) -> R<&'i Operand> {
    i.args.get(n).ok_or_else(|| format!("line {}: {} is missing operand {}", i.line, i.op, n + 1))
}

/// Expand `%r<4>` declarations into `%r0..%r3`.
fn decl_names(v: &VarDecl) -> Vec<String> {
    match v.count {
        Some(n) => (0..n).map(|i| format!("{}{}", v.name, i)).collect(),
        None => vec![v.name.clone()],
    }
}

fn collect_labels(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Label(l) => out.push(l.clone()),
            Stmt::Block(b) => collect_labels(b, out),
            _ => {}
        }
    }
}

impl<'a> Compiler<'a> {
    // ------------------------------------------------------------------ types

    fn scalar_ty(&mut self, s: Scalar) -> Id {
        match s {
            Scalar::Pred => self.b.bool(),
            Scalar::F16 => self.b.float(16),
            Scalar::F32 => self.b.float(32),
            Scalar::F64 => self.b.float(64),
            Scalar::BF16 => self.b.int(16),
            Scalar::B128 => {
                let i = self.b.int(64);
                self.b.vector(i, 2)
            }
            s => self.b.int(s.size() * 8),
        }
    }

    fn value_ty(&mut self, t: ptx::Type) -> Id {
        let s = self.scalar_ty(t.scalar);
        if t.vector > 0 {
            self.b.vector(s, t.vector as u32)
        } else {
            s
        }
    }

    /// In-memory type of a declared variable (arrays flattened).
    fn mem_ty(&mut self, v: &VarDecl) -> Id {
        let mut t = if v.ty.scalar == Scalar::Pred {
            self.b.int(8)
        } else {
            self.value_ty(v.ty)
        };
        if !v.dims.is_empty() {
            let n = v.elements().max(1) as u32;
            t = self.b.array(t, n);
        }
        t
    }

    fn is_float_ty(&self, t: Id) -> bool {
        matches!(self.b.type_of(t), TypeKey::Float(_))
    }

    fn is_bool_ty(&self, t: Id) -> bool {
        matches!(self.b.type_of(t), TypeKey::Bool)
    }

    // -------------------------------------------------------- module globals

    fn declare_global(&mut self, v: &VarDecl) -> R<()> {
        match v.space {
            StateSpace::Global | StateSpace::Const | StateSpace::Shared => {}
            s => return Err(format!("line {}: module-scope .{s:?} variables are not supported", v.line)),
        }
        if v.ty.scalar == Scalar::TexRef || v.ty.scalar == Scalar::SurfRef || v.ty.scalar == Scalar::SamplerRef {
            return Err(format!("line {}: texture references are not supported yet", v.line));
        }
        let storage = storage_of(v.space);
        if v.space == StateSpace::Shared && v.dims.first() == Some(&None) {
            // extern .shared .b8 smem[] -> dynamic shared memory
            self.has_dyn_shared = true;
            self.globals.insert(v.name.clone(), Sym { ptr: 0, storage, dyn_shared: true });
            return Ok(());
        }
        let ty = self.mem_ty(v);
        let pty = self.b.ptr(storage, ty);
        let init = if v.space == StateSpace::Shared {
            None
        } else {
            Some(match &v.init {
                Some(i) => self.initializer(v, ty, i)?,
                None => self.b.const_null(ty),
            })
        };
        let var = self.b.global_var(pty, storage, init);
        self.b.name(var, &v.name);
        if let Some(a) = v.align {
            self.b.decorate(var, deco::ALIGNMENT, &[a]);
        }
        if v.space == StateSpace::Shared {
            self.static_shared += v.byte_size();
        } else {
            self.global_list.push((
                GlobalInfo { name: v.name.clone(), size: v.byte_size(), is_const: v.space == StateSpace::Const },
                var,
            ));
        }
        self.globals.insert(v.name.clone(), Sym { ptr: var, storage, dyn_shared: false });
        Ok(())
    }

    fn initializer(&mut self, v: &VarDecl, ty: Id, init: &ptx::Init) -> R<Id> {
        let elem = self.value_ty(v.ty);
        let mut flat = Vec::new();
        fn walk(i: &ptx::Init, out: &mut Vec<ptx::Expr>) {
            match i {
                ptx::Init::Expr(e) => out.push(e.clone()),
                ptx::Init::List(l) => l.iter().for_each(|x| walk(x, out)),
            }
        }
        walk(init, &mut flat);
        if v.ty.vector > 0 {
            return Err(format!("line {}: vector-typed initializers are not supported", v.line));
        }
        let mut consts = Vec::new();
        for e in &flat {
            let bits = match e {
                ptx::Expr::Int(x) => {
                    if v.ty.scalar.is_float() {
                        match v.ty.scalar {
                            Scalar::F64 => (*x as f64).to_bits(),
                            _ => (*x as f32).to_bits() as u64,
                        }
                    } else {
                        *x as u64
                    }
                }
                ptx::Expr::F32(b) => match v.ty.scalar {
                    Scalar::F64 => (f32::from_bits(*b) as f64).to_bits(),
                    _ => *b as u64,
                },
                ptx::Expr::F64(b) => match v.ty.scalar {
                    Scalar::F32 => (f64::from_bits(*b) as f32).to_bits() as u64,
                    _ => *b,
                },
                ptx::Expr::Sym { name, .. } => {
                    return Err(format!(
                        "line {}: initializer referencing symbol {name} is not supported yet",
                        v.line
                    ))
                }
            };
            consts.push(self.b.const_bits(elem, bits));
        }
        if v.dims.is_empty() {
            return Ok(consts.first().copied().unwrap_or_else(|| self.b.const_null(elem)));
        }
        let n = v.elements() as usize;
        let zero = self.b.const_bits(elem, 0);
        consts.resize(n, zero);
        Ok(self.b.const_composite(ty, consts))
    }

    // ------------------------------------------------------ function headers

    fn param_ty(&mut self, p: &VarDecl) -> Id {
        if p.dims.is_empty() && p.ty.vector == 0 {
            if p.ty.scalar == Scalar::Pred {
                return self.b.int(8);
            }
            self.scalar_ty(p.ty.scalar)
        } else {
            self.mem_ty(p)
        }
    }

    fn declare_func(&mut self, f: &Function) -> R<()> {
        if self.funcs.contains_key(&f.name) {
            return Ok(()); // a prior declaration
        }
        let id = self.b.id();
        let mut params = Vec::new();
        for p in &f.params {
            let t = if f.kind == FuncKind::Entry && !p.dims.is_empty() {
                // by-value aggregate kernel argument
                let a = self.mem_ty(p);
                self.b.ptr(Storage::Function, a)
            } else {
                self.param_ty(p)
            };
            params.push((p.clone(), t));
        }
        let mut rets = Vec::new();
        for r in &f.ret {
            let t = self.param_ty(r);
            rets.push((r.clone(), t));
        }
        let ret_ty = match rets.len() {
            0 => self.b.void(),
            1 => rets[0].1,
            _ => {
                let ms = rets.iter().map(|r| r.1).collect();
                self.b.ty(TypeKey::Struct(ms))
            }
        };
        self.funcs.insert(f.name.clone(), FuncSig { id, ret_ty, rets, params });
        if f.body.is_none() {
            if f.linkage == Linkage::Extern && !is_runtime_builtin(&f.name) {
                return Err(format!("external function {} is not supported", f.name));
            }
        }
        Ok(())
    }

    fn fn_type_of(&mut self, name: &str, entry: bool) -> (Id, Vec<Id>) {
        let sig = &self.funcs[name];
        let ret = sig.ret_ty;
        let mut ps: Vec<Id> = sig.params.iter().map(|p| p.1).collect();
        if self.has_dyn_shared {
            let i8t = self.b.int(8);
            let wp = self.b.ptr(Storage::Workgroup, i8t);
            ps.push(wp);
        }
        let _ = entry;
        let ft = self.b.fn_type(ret, ps.clone());
        (ft, ps)
    }

    // --------------------------------------------------------- function body

    fn function(&mut self, f: &Function) -> R<Option<KernelInfo>> {
        let is_entry = f.kind == FuncKind::Entry;
        let (ft, ptys) = self.fn_type_of(&f.name, is_entry);
        let fid = self.funcs[&f.name].id;
        let ret_ty = self.funcs[&f.name].ret_ty;
        self.b.name(fid, &f.name);

        self.f = FnState { is_entry, ret_ty, ..Default::default() };
        self.f.scopes.push(HashMap::new());

        let saved = std::mem::take(&mut self.b.functions);
        // OpFunction + parameters
        self.b.emit(op::FUNCTION, &[ret_ty, fid, 0, ft]);
        let mut param_ids = Vec::new();
        for &t in &ptys {
            let p = self.b.id();
            self.b.emit(op::FUNCTION_PARAMETER, &[t, p]);
            param_ids.push(p);
        }
        let header = std::mem::take(&mut self.b.functions);

        let entry_label = self.b.id();
        self.f.cur_block = entry_label;
        self.b.emit(op::LABEL, &[entry_label]);

        let mut labels = Vec::new();
        collect_labels(f.body.as_ref().unwrap(), &mut labels);
        for l in labels {
            let id = self.b.id();
            self.f.labels.insert(l, id);
        }

        // Parameters -> local copies.
        let sig_params: Vec<(VarDecl, Id)> = self.funcs[&f.name].params.clone();
        let mut pinfo = Vec::new();
        for (i, (p, t)) in sig_params.iter().enumerate() {
            let pid = param_ids[i];
            if is_entry {
                pinfo.push(ParamInfo {
                    name: p.name.clone(),
                    size: p.byte_size() as u32,
                    align: p.align.unwrap_or(p.ty.size()).max(1),
                });
            }
            if is_entry && !p.dims.is_empty() {
                self.b.decorate(pid, deco::FUNC_PARAM_ATTR, &[2 /* ByVal */]);
                self.b.decorate(pid, deco::ALIGNMENT, &[p.align.unwrap_or(8).max(8)]);
                self.bind(&p.name, Var::Mem(Sym { ptr: pid, storage: Storage::Function, dyn_shared: false }));
                continue;
            }
            if p.space == StateSpace::Reg {
                let var = self.local(*t);
                self.b.emit(op::STORE, &[var, pid]);
                self.bind(&p.name, Var::Reg { var, ty: *t, scalar: p.ty.scalar, vector: p.ty.vector });
            } else {
                let var = self.local(*t);
                self.b.emit(op::STORE, &[var, pid]);
                self.bind(&p.name, Var::Mem(Sym { ptr: var, storage: Storage::Function, dyn_shared: false }));
            }
        }
        if self.has_dyn_shared {
            self.f.dyn_smem = Some(*param_ids.last().unwrap());
        }
        // Return values of .func.
        let sig_rets: Vec<(VarDecl, Id)> = self.funcs[&f.name].rets.clone();
        for (r, t) in &sig_rets {
            let var = self.local(*t);
            let v = if r.space == StateSpace::Reg {
                Var::Reg { var, ty: *t, scalar: r.ty.scalar, vector: r.ty.vector }
            } else {
                Var::Mem(Sym { ptr: var, storage: Storage::Function, dyn_shared: false })
            };
            self.bind(&r.name, v.clone());
            self.f.rets.push((v, *t));
        }

        self.stmts(f.body.as_ref().unwrap())?;
        if !self.f.terminated {
            self.ret()?;
        }
        self.b.emit(op::FUNCTION_END, &[]);

        // Assemble: header, entry label, local variables, body.
        let body = std::mem::take(&mut self.b.functions);
        let vars = std::mem::take(&mut self.b.local_vars);
        self.b.functions = saved;
        self.b.functions.extend(header);
        self.b.functions.extend(&body[..2]); // OpLabel entry
        self.b.functions.extend(vars);
        self.b.functions.extend(&body[2..]);

        if !is_entry {
            return Ok(None);
        }
        self.entry_list.push((fid, f.name.clone()));
        if let Some(r) = f.reqntid {
            self.b.execution_mode(fid, 17 /* LocalSize */, &r);
        }
        if self.f.uses_warp_ops && f.reqntid.is_none() {
            self.b.capability(cap::SUBGROUP_DISPATCH);
            self.b.execution_mode(fid, 35 /* SubgroupSize */, &[self.opts.warp_size]);
        }
        Ok(Some(KernelInfo {
            name: f.name.clone(),
            dyn_shared_arg: self.has_dyn_shared.then_some(pinfo.len() as u32),
            params: pinfo,
            reqntid: f.reqntid,
            maxntid: f.maxntid,
            static_shared: self.static_shared,
            uses_warp_ops: self.f.uses_warp_ops,
        }))
    }

    /// Synthesize `kernel void __ignuda_global_addresses(ulong *out)`.
    fn globals_kernel(&mut self) {
        let void = self.b.void();
        let u64t = self.b.int(64);
        let pout = self.b.ptr(Storage::CrossWorkgroup, u64t);
        let ft = self.b.fn_type(void, vec![pout]);
        let fid = self.b.id();
        let p = self.b.id();
        let l = self.b.id();
        self.b.emit(op::FUNCTION, &[void, fid, 0, ft]);
        self.b.emit(op::FUNCTION_PARAMETER, &[pout, p]);
        self.b.emit(op::LABEL, &[l]);
        let list: Vec<Id> = self.global_list.iter().map(|g| g.1).collect();
        for (i, g) in list.into_iter().enumerate() {
            let idx = self.b.const_u64(i as u64);
            let slot = self.b.op(op::PTR_ACCESS_CHAIN, pout, &[p, idx]);
            let a = self.b.op(op::CONVERT_PTR_TO_U, u64t, &[g]);
            self.b.emit(op::STORE, &[slot, a]);
        }
        self.b.emit(op::RETURN, &[]);
        self.b.emit(op::FUNCTION_END, &[]);
        self.entry_list.push((fid, GLOBALS_KERNEL.to_string()));
    }

    fn local(&mut self, t: Id) -> Id {
        let pt = self.b.ptr(Storage::Function, t);
        self.b.local_var(pt)
    }

    fn bind(&mut self, name: &str, v: Var) {
        self.f.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn lookup(&self, name: &str) -> Option<Var> {
        for s in self.f.scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        self.globals.get(name).map(|s| Var::Mem(s.clone()))
    }

    fn sym_ptr(&self, s: &Sym) -> R<Id> {
        if s.dyn_shared {
            self.f.dyn_smem.ok_or_else(|| "dynamic shared memory used without hidden argument".to_string())
        } else {
            Ok(s.ptr)
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) -> R<()> {
        for s in stmts {
            match s {
                Stmt::Decl(v) => self.local_decl(v)?,
                Stmt::Label(l) => {
                    let id = self.f.labels[l];
                    if !self.f.terminated {
                        self.b.emit(op::BRANCH, &[id]);
                    }
                    self.b.emit(op::LABEL, &[id]);
                    self.f.cur_block = id;
                    self.f.terminated = false;
                }
                Stmt::Block(b) => {
                    self.f.scopes.push(HashMap::new());
                    self.stmts(b)?;
                    self.f.scopes.pop();
                }
                Stmt::CallProto { name, .. } => {
                    self.f.protos.insert(name.clone(), ());
                }
                Stmt::Inst(i) => {
                    if self.f.terminated {
                        // unreachable code after a branch: give it a block
                        let l = self.b.id();
                        self.b.emit(op::LABEL, &[l]);
                        self.f.cur_block = l;
                        self.f.terminated = false;
                    }
                    self.inst(i).map_err(|e| {
                        if e.starts_with("line") {
                            e
                        } else {
                            format!("line {}: {}{}: {e}", i.line, i.op, i.mods.iter().map(|m| format!(".{m}")).collect::<String>())
                        }
                    })?;
                }
            }
        }
        Ok(())
    }

    fn local_decl(&mut self, v: &VarDecl) -> R<()> {
        match v.space {
            StateSpace::Reg => {
                let ty = self.value_ty(v.ty);
                for n in decl_names(v) {
                    let var = self.local(ty);
                    self.bind(&n, Var::Reg { var, ty, scalar: v.ty.scalar, vector: v.ty.vector });
                }
            }
            StateSpace::Local | StateSpace::Param => {
                let ty = self.mem_ty(v);
                let var = self.local(ty);
                if let Some(a) = v.align {
                    self.b.decorate(var, deco::ALIGNMENT, &[a]);
                }
                self.bind(&v.name, Var::Mem(Sym { ptr: var, storage: Storage::Function, dyn_shared: false }));
            }
            StateSpace::Shared => {
                // function-scope shared: hoist to module scope
                let ty = self.mem_ty(v);
                let pty = self.b.ptr(Storage::Workgroup, ty);
                let var = self.b.global_var(pty, Storage::Workgroup, None);
                if let Some(a) = v.align {
                    self.b.decorate(var, deco::ALIGNMENT, &[a]);
                }
                self.static_shared += v.byte_size();
                self.bind(&v.name, Var::Mem(Sym { ptr: var, storage: Storage::Workgroup, dyn_shared: false }));
            }
            s => return Err(format!("line {}: .{s:?} declaration inside a function is not supported", v.line)),
        }
        Ok(())
    }

    // ------------------------------------------------------ value plumbing

    /// Convert value `v` of type `from` to type `to`. Width changes extend
    /// with sign if `signed`; same-width changes are bit casts.
    fn coerce(&mut self, v: Id, from: Id, to: Id, signed: bool) -> Id {
        if from == to {
            return v;
        }
        let fb = self.is_bool_ty(from);
        let tb = self.is_bool_ty(to);
        if fb {
            let rt = if self.is_float_ty(to) {
                let w = self.b.width(to);
                self.b.int(w)
            } else {
                to
            };
            let one = self.b.const_bits(rt, 1);
            let zero = self.b.const_bits(rt, 0);
            let r = self.b.op(op::SELECT, rt, &[v, one, zero]);
            return self.coerce(r, rt, to, false);
        }
        if tb {
            let w = self.b.width(from);
            let it = self.b.int(w);
            let iv = self.coerce(v, from, it, false);
            let zero = self.b.const_bits(it, 0);
            return self.b.op(op::I_NOT_EQUAL, to, &[iv, zero]);
        }
        let fw = self.b.width(from);
        let tw = self.b.width(to);
        if fw == tw {
            return self.b.op(op::BITCAST, to, &[v]);
        }
        // different widths: go through integers
        let fi = self.b.int(fw);
        let ti = self.b.int(tw);
        let a = if fi == from { v } else { self.b.op(op::BITCAST, fi, &[v]) };
        let c = if matches!(self.b.type_of(from), TypeKey::Vector(..)) || matches!(self.b.type_of(to), TypeKey::Vector(..)) {
            // vector <-> scalar of different size: not expected
            a
        } else if tw > fw && signed {
            self.b.op(op::S_CONVERT, ti, &[a])
        } else {
            self.b.op(op::U_CONVERT, ti, &[a])
        };
        if ti == to {
            c
        } else {
            self.b.op(op::BITCAST, to, &[c])
        }
    }

    fn special_reg(&mut self, name: &str, comp: Option<u8>) -> R<(Id, Id)> {
        let u32t = self.b.int(32);
        let u64t = self.b.int(64);
        let vec3 = |c: &mut Self, bi: u32| -> Id {
            let v3 = c.b.vector(u64t, 3);
            let var = c.builtin(bi, v3);
            let v = c.b.op(op::LOAD, v3, &[var]);
            let e = c.b.op(op::COMPOSITE_EXTRACT, u64t, &[v, comp.unwrap_or(0) as u32]);
            c.b.op(op::U_CONVERT, u32t, &[e])
        };
        let v = match name {
            "%tid" => vec3(self, builtin::LOCAL_INVOCATION_ID),
            "%ntid" => vec3(self, builtin::WORKGROUP_SIZE),
            "%ctaid" => vec3(self, builtin::WORKGROUP_ID),
            "%nctaid" => vec3(self, builtin::NUM_WORKGROUPS),
            "%laneid" => {
                self.f.uses_warp_ops = true;
                let var = self.builtin(builtin::SUBGROUP_LOCAL_INVOCATION_ID, u32t);
                self.b.op(op::LOAD, u32t, &[var])
            }
            "%warpid" => {
                self.f.uses_warp_ops = true;
                let var = self.builtin(builtin::SUBGROUP_ID, u32t);
                self.b.op(op::LOAD, u32t, &[var])
            }
            "%lanemask_eq" | "%lanemask_lt" | "%lanemask_le" | "%lanemask_gt" | "%lanemask_ge" => {
                self.f.uses_warp_ops = true;
                let var = self.builtin(builtin::SUBGROUP_LOCAL_INVOCATION_ID, u32t);
                let lane = self.b.op(op::LOAD, u32t, &[var]);
                let one = self.b.const_u32(1);
                let bit = self.b.op(op::SHIFT_LEFT_LOGICAL, u32t, &[one, lane]);
                let lt = self.b.op(op::I_SUB, u32t, &[bit, one]);
                match name {
                    "%lanemask_eq" => bit,
                    "%lanemask_lt" => lt,
                    "%lanemask_le" => self.b.op(op::BITWISE_OR, u32t, &[lt, bit]),
                    "%lanemask_ge" => self.b.op(op::NOT, u32t, &[lt]),
                    _ => {
                        let le = self.b.op(op::BITWISE_OR, u32t, &[lt, bit]);
                        self.b.op(op::NOT, u32t, &[le])
                    }
                }
            }
            "%warpsize" | "WARP_SZ" => self.b.const_u32(self.opts.warp_size),
            "%smid" | "%gridid" => self.b.const_u32(0),
            "%nsmid" => self.b.const_u32(1),
            "%clock" | "%clock64" | "%globaltimer" | "%globaltimer_lo" => {
                // TODO: OpReadClockKHR when the device supports cl_khr_kernel_clock
                if name == "%clock64" || name == "%globaltimer" {
                    return Ok((self.b.const_u64(0), u64t));
                }
                self.b.const_u32(0)
            }
            _ => return Err(format!("unknown special register {name}")),
        };
        Ok((v, u32t))
    }

    fn builtin(&mut self, bi: u32, ty: Id) -> Id {
        if let Some(&v) = self.builtins.get(&bi) {
            return v;
        }
        let pt = self.b.ptr(Storage::Input, ty);
        let v = self.b.global_var(pt, Storage::Input, None);
        self.b.decorate(v, deco::BUILTIN, &[bi]);
        self.b.decorate(v, deco::CONSTANT, &[]);
        self.builtins.insert(bi, v);
        v
    }

    /// Immediate of type `ty` from an integer literal.
    fn imm_int(&mut self, x: i128, ty: Id) -> Id {
        match self.b.type_of(ty).clone() {
            TypeKey::Float(64) => self.b.const_bits(ty, (x as f64).to_bits()),
            TypeKey::Float(32) => self.b.const_bits(ty, (x as f32).to_bits() as u64),
            TypeKey::Float(16) => self.b.const_bits(ty, f32_to_f16(x as f32) as u64),
            _ => self.b.const_bits(ty, x as u64),
        }
    }

    fn imm_float(&mut self, v: f64, bits32: Option<u32>, bits64: Option<u64>, ty: Id) -> Id {
        match self.b.type_of(ty).clone() {
            TypeKey::Float(64) => self.b.const_bits(ty, bits64.unwrap_or_else(|| v.to_bits())),
            TypeKey::Float(32) => self.b.const_bits(ty, bits32.unwrap_or_else(|| (v as f32).to_bits()) as u64),
            TypeKey::Float(16) => self.b.const_bits(ty, f32_to_f16(v as f32) as u64),
            // float literal used as raw bits for an integer type
            _ => self.b.const_bits(ty, bits32.map(|b| b as u64).or(bits64).unwrap_or(v.to_bits())),
        }
    }

    /// Read operand `o` as a value of SPIR-V type `ty` (`signed` governs
    /// sign extension when a narrower register is read).
    fn read(&mut self, o: &Operand, ty: Id, signed: bool) -> R<Id> {
        match o {
            Operand::Int(x) => Ok(self.imm_int(*x, ty)),
            Operand::F32(b) => Ok(self.imm_float(f32::from_bits(*b) as f64, Some(*b), None, ty)),
            Operand::F64(b) => Ok(self.imm_float(f64::from_bits(*b), None, Some(*b), ty)),
            Operand::Not(inner) => {
                let v = self.read(inner, ty, signed)?;
                if self.is_bool_ty(ty) {
                    Ok(self.b.op(op::LOGICAL_NOT, ty, &[v]))
                } else {
                    Ok(self.b.op(op::NOT, ty, &[v]))
                }
            }
            Operand::Id { name, comp } => {
                if name.starts_with('%') && self.lookup(name).is_none() || comp.is_some() && self.lookup(name).is_none() {
                    let (v, vt) = self.special_reg(name, *comp)?;
                    return Ok(self.coerce(v, vt, ty, false));
                }
                match self.lookup(name) {
                    Some(Var::Reg { var, ty: rt, .. }) => {
                        let v = self.b.op(op::LOAD, rt, &[var]);
                        if let Some(c) = comp {
                            // component of a vector register
                            if let TypeKey::Vector(e, _) = self.b.type_of(rt).clone() {
                                let x = self.b.op(op::COMPOSITE_EXTRACT, e, &[v, *c as u32]);
                                return Ok(self.coerce(x, e, ty, signed));
                            }
                        }
                        Ok(self.coerce(v, rt, ty, signed))
                    }
                    Some(Var::Mem(s)) => {
                        // address of a variable
                        let p = self.sym_ptr(&s)?;
                        let w = self.b.width(ty);
                        let it = self.b.int(w);
                        let a = if s.storage == Storage::Workgroup {
                            let off = self.shared_ptr_to_int(p);
                            let u64t = self.b_int64();
                            self.coerce(off, u64t, it, false)
                        } else {
                            self.b.op(op::CONVERT_PTR_TO_U, it, &[p])
                        };
                        Ok(self.coerce(a, it, ty, false))
                    }
                    None => Err(format!("unknown identifier {name}")),
                }
            }
            Operand::SymOff { name, offset } => {
                let base = self.read(&Operand::Id { name: name.clone(), comp: None }, ty, false)?;
                let off = self.b.const_bits(ty, *offset as u64);
                Ok(self.b.op(op::I_ADD, ty, &[base, off]))
            }
            Operand::Vector(parts) => {
                // pack {a, b, ...} into one value
                let w = self.b.width(ty);
                let n = parts.len() as u32;
                let et = self.b.int(w / n);
                let mut vals = Vec::new();
                for p in parts {
                    vals.push(self.read(p, et, false)?);
                }
                let vt = self.b.vector(et, n);
                let v = self.b.op(op::COMPOSITE_CONSTRUCT, vt, &vals);
                Ok(self.b.op(op::BITCAST, ty, &[v]))
            }
            _ => Err(format!("unsupported operand {o:?}")),
        }
    }

    /// Write value `v` (type `vt`) to destination operand `d`.
    fn write(&mut self, d: &Operand, v: Id, vt: Id, signed: bool) -> R<()> {
        match d {
            Operand::Id { name, comp } => {
                if name == "_" {
                    return Ok(());
                }
                match self.lookup(name) {
                    Some(Var::Reg { var, ty, .. }) => {
                        if let Some(c) = comp {
                            if let TypeKey::Vector(e, _) = self.b.type_of(ty).clone() {
                                let cur = self.b.op(op::LOAD, ty, &[var]);
                                let x = self.coerce(v, vt, e, signed);
                                let nv = self.b.op(op::COMPOSITE_INSERT, ty, &[x, cur, *c as u32]);
                                self.b.emit(op::STORE, &[var, nv]);
                                return Ok(());
                            }
                        }
                        let x = self.coerce(v, vt, ty, signed);
                        self.b.emit(op::STORE, &[var, x]);
                        Ok(())
                    }
                    _ => Err(format!("cannot write to {name}")),
                }
            }
            Operand::Vector(parts) => {
                // unpack into {a, b, ...}
                let w = self.b.width(vt);
                let n = parts.len() as u32;
                let et = self.b.int(w / n);
                let vect = self.b.vector(et, n);
                let vv = self.b.op(op::BITCAST, vect, &[v]);
                for (k, p) in parts.iter().enumerate() {
                    let x = self.b.op(op::COMPOSITE_EXTRACT, et, &[vv, k as u32]);
                    self.write(p, x, et, false)?;
                }
                Ok(())
            }
            _ => Err(format!("unsupported destination {d:?}")),
        }
    }

    /// Typed pointer for a memory operand.
    fn address(&mut self, a: &Operand, space: StateSpace, elem: Id) -> R<Id> {
        let (base, offset) = match a {
            Operand::Addr { base, offset } => (base.clone(), *offset),
            Operand::Id { name, comp: None } => (Some(name.clone()), 0),
            _ => return Err(format!("bad address operand {a:?}")),
        };
        let u64t = self.b.int(64);
        let i8t = self.b.int(8);
        match base.as_deref().map(|n| (n, self.lookup(n))) {
            Some((_, Some(Var::Mem(s)))) => {
                let p = self.sym_ptr(&s)?;
                let bp = self.b.ptr(s.storage, i8t);
                let mut p = self.b.op(op::BITCAST, bp, &[p]);
                if offset != 0 {
                    let off = self.b.const_u64(offset as u64);
                    p = self.b.op(op::IN_BOUNDS_PTR_ACCESS_CHAIN, bp, &[p, off]);
                }
                let target = if space == StateSpace::Generic { Storage::Generic } else { s.storage };
                if target == Storage::Generic && s.storage != Storage::Generic {
                    let gp = self.b.ptr(Storage::Generic, i8t);
                    p = self.b.op(op::PTR_CAST_TO_GENERIC, gp, &[p]);
                }
                let tp = self.b.ptr(target, elem);
                Ok(self.b.op(op::BITCAST, tp, &[p]))
            }
            Some((name, Some(Var::Reg { .. }))) => {
                let shared = space == StateSpace::Shared;
                let mut v = self.read(&Operand::Id { name: name.to_string(), comp: None }, u64t, shared)?;
                if offset != 0 {
                    let off = self.b.const_u64(offset as u64);
                    v = self.b.op(op::I_ADD, u64t, &[v, off]);
                }
                let tp = self.b.ptr(storage_of(space), elem);
                if shared {
                    let p = self.shared_int_to_ptr(v);
                    return Ok(self.b.op(op::BITCAST, tp, &[p]));
                }
                Ok(self.b.op(op::CONVERT_U_TO_PTR, tp, &[v]))
            }
            Some((name, None)) => Err(format!("unknown address base {name}")),
            None => {
                let v = self.b.const_u64(offset as u64);
                let tp = self.b.ptr(storage_of(space), elem);
                if space == StateSpace::Shared {
                    let p = self.shared_int_to_ptr(v);
                    return Ok(self.b.op(op::BITCAST, tp, &[p]));
                }
                Ok(self.b.op(op::CONVERT_U_TO_PTR, tp, &[v]))
            }
        }
    }

    fn b_int64(&mut self) -> Id {
        self.b.int(64)
    }

    fn anchor(&mut self) -> Id {
        if let Some(a) = self.shared_anchor {
            return a;
        }
        let u32t = self.b.int(32);
        let pt = self.b.ptr(Storage::Workgroup, u32t);
        let v = self.b.global_var(pt, Storage::Workgroup, None);
        self.b.name(v, "__ignuda_shared_anchor");
        self.b.decorate(v, deco::ALIGNMENT, &[16]);
        self.shared_anchor = Some(v);
        v
    }

    /// Integer (u64) shared-window address of a Workgroup pointer: its offset
    /// from the anchor variable. Shared addresses are often held in 32-bit
    /// registers, which cannot hold a raw pointer on every device.
    fn shared_ptr_to_int(&mut self, p: Id) -> Id {
        let u64t = self.b.int(64);
        let a = self.anchor();
        let ai = self.b.op(op::CONVERT_PTR_TO_U, u64t, &[a]);
        let pi = self.b.op(op::CONVERT_PTR_TO_U, u64t, &[p]);
        self.b.op(op::I_SUB, u64t, &[pi, ai])
    }

    /// Workgroup i8 pointer for a u64 shared-window address.
    fn shared_int_to_ptr(&mut self, v: Id) -> Id {
        let u64t = self.b.int(64);
        let i8t = self.b.int(8);
        let a = self.anchor();
        let ai = self.b.op(op::CONVERT_PTR_TO_U, u64t, &[a]);
        let x = self.b.op(op::I_ADD, u64t, &[ai, v]);
        let pt = self.b.ptr(Storage::Workgroup, i8t);
        self.b.op(op::CONVERT_U_TO_PTR, pt, &[x])
    }

    fn ty_of(&mut self, s: Scalar) -> Id {
        self.scalar_ty(s)
    }

    // ------------------------------------------------------- control flow

    fn new_block(&mut self) -> Id {
        let l = self.b.id();
        self.b.emit(op::LABEL, &[l]);
        self.f.cur_block = l;
        self.f.terminated = false;
        l
    }

    fn ret(&mut self) -> R<()> {
        if self.f.rets.is_empty() {
            self.b.emit(op::RETURN, &[]);
        } else {
            let mut vals = Vec::new();
            for (v, t) in self.f.rets.clone() {
                let ptr = match v {
                    Var::Reg { var, .. } => var,
                    Var::Mem(s) => s.ptr,
                };
                vals.push(self.b.op(op::LOAD, t, &[ptr]));
            }
            let r = if vals.len() == 1 {
                vals[0]
            } else {
                let rt = self.f.ret_ty;
                self.b.op(op::COMPOSITE_CONSTRUCT, rt, &vals)
            };
            self.b.emit(op::RETURN_VALUE, &[r]);
        }
        self.f.terminated = true;
        Ok(())
    }

    fn pred_value(&mut self, neg: bool, name: &str) -> R<Id> {
        let bt = self.b.bool();
        let v = self.read(&Operand::Id { name: name.to_string(), comp: None }, bt, false)?;
        Ok(if neg { self.b.op(op::LOGICAL_NOT, bt, &[v]) } else { v })
    }

    fn inst(&mut self, i: &Inst) -> R<()> {
        if let Some((neg, p)) = &i.pred {
            let cond = self.pred_value(*neg, p)?;
            if i.op == "bra" {
                let target = self.label(arg(i, 0)?)?;
                let next = self.b.id();
                self.b.emit(op::BRANCH_CONDITIONAL, &[cond, target, next]);
                self.b.emit(op::LABEL, &[next]);
                self.f.cur_block = next;
                return Ok(());
            }
            let then = self.b.id();
            let merge = self.b.id();
            self.b.emit(op::BRANCH_CONDITIONAL, &[cond, then, merge]);
            self.b.emit(op::LABEL, &[then]);
            self.f.cur_block = then;
            self.f.terminated = false;
            self.inst_body(i)?;
            if !self.f.terminated {
                self.b.emit(op::BRANCH, &[merge]);
            }
            self.b.emit(op::LABEL, &[merge]);
            self.f.cur_block = merge;
            self.f.terminated = false;
            return Ok(());
        }
        self.inst_body(i)
    }

    fn label(&mut self, o: &Operand) -> R<Id> {
        let n = o.id().ok_or("branch target must be a label")?;
        self.f.labels.get(n).copied().ok_or_else(|| format!("unknown label {n}"))
    }

    fn inst_body(&mut self, i: &Inst) -> R<()> {
        match i.op.as_str() {
            "bra" | "brx" => {
                if i.op == "brx" {
                    return Err("brx.idx is not supported yet".into());
                }
                let t = self.label(arg(i, 0)?)?;
                self.b.emit(op::BRANCH, &[t]);
                self.f.terminated = true;
            }
            "ret" => self.ret()?,
            "exit" | "trap" => {
                // `exit` in a .func terminates the thread; returning is the
                // closest approximation without a thread-exit primitive.
                if self.f.rets.is_empty() {
                    self.b.emit(op::RETURN, &[]);
                    self.f.terminated = true;
                } else {
                    self.ret()?;
                }
            }
            "mov" => self.mov(i)?,
            "ld" | "ldu" => self.ld(i)?,
            "st" => self.st(i)?,
            "cvta" => self.cvta(i)?,
            "cvt" => self.cvt(i)?,
            "setp" => self.setp(i)?,
            "selp" => {
                let t = inst_type(i)?;
                let ty = self.ty_of(t);
                let a = self.read(arg(i, 1)?, ty, t.is_signed())?;
                let b = self.read(arg(i, 2)?, ty, t.is_signed())?;
                let bt = self.b.bool();
                let c = self.read(arg(i, 3)?, bt, false)?;
                let r = self.b.op(op::SELECT, ty, &[c, a, b]);
                self.write(arg(i, 0)?, r, ty, t.is_signed())?;
            }
            "bar" | "barrier" => {
                if i.has("arrive") {
                    return Ok(()); // TODO: split arrive/wait barriers
                }
                if i.has("red") {
                    return self.bar_red(i);
                }
                self.barrier();
            }
            "membar" | "fence" => {
                let scope = if i.has("cta") { 2 } else { 1 };
                let sc = self.b.const_u32(scope);
                let sem = self.b.const_u32(0x8 | 0x100 | 0x200);
                self.b.emit(op::MEMORY_BARRIER, &[sc, sem]);
            }
            "atom" | "red" => self.atom(i)?,
            "call" => self.call(i)?,
            "add" | "sub" | "mul" | "div" | "rem" | "min" | "max" | "and" | "or" | "xor" | "shl" | "shr" | "addc"
            | "subc" | "mul24" | "copysign" => self.binary(i)?,
            "mad" | "fma" | "mad24" | "madc" => self.mad(i)?,
            "abs" | "neg" | "not" | "cnot" | "popc" | "clz" | "brev" | "rcp" | "sqrt" | "rsqrt" | "sin" | "cos"
            | "lg2" | "ex2" | "tanh" => self.unary(i)?,
            "testp" => self.testp(i)?,
            "bfe" => self.bfe(i)?,
            "bfi" => self.bfi(i)?,
            "prmt" => self.prmt(i)?,
            "shf" => self.shf(i)?,
            "prefetch" | "prefetchu" | "pmevent" | "nanosleep" | "discard" | "applypriority" | "createpolicy" => {
                if i.op == "createpolicy" {
                    let u64t = self.b.int(64);
                    let z = self.b.const_u64(0);
                    self.write(arg(i, 0)?, z, u64t, false)?;
                }
            }
            "activemask" => {
                self.f.uses_warp_ops = true;
                let u32t = self.b.int(32);
                let t = self.b.const_bool(true);
                let m = self.ballot(t)?;
                self.write(arg(i, 0)?, m, u32t, false)?;
            }
            "vote" => self.vote(i)?,
            "shfl" => self.shfl(i)?,
            op => return Err(format!("unsupported instruction {op}")),
        }
        Ok(())
    }

    fn barrier(&mut self) {
        let wg = self.b.const_u32(2);
        let sem = self.b.const_u32(0x10 | 0x100 | 0x200);
        self.b.emit(op::CONTROL_BARRIER, &[wg, wg, sem]);
    }

    // ------------------------------------------------------------- moves

    fn mov(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let src = arg(i, 1)?;
        let dst = arg(i, 0)?;
        // function address: mov.u64 %rd, func
        if let Some(n) = src.id() {
            if self.funcs.contains_key(n) && self.lookup(n).is_none() {
                return Err("taking the address of a function is not supported yet".into());
            }
        }
        let v = self.read(src, ty, t.is_signed())?;
        self.write(dst, v, ty, t.is_signed())
    }

    fn cvta(&mut self, i: &Inst) -> R<()> {
        // cvta.space.u64 d, a  (to generic) / cvta.to.space.u64 d, a
        let to = i.has("to");
        let space = space_of(i);
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let a = arg(i, 1)?;
        let v = self.read(a, ty, false)?;
        let r = match space {
            // Global and generic addresses coincide on Intel GPUs and CPUs.
            StateSpace::Global | StateSpace::Const => v,
            StateSpace::Shared | StateSpace::Local | StateSpace::Param => {
                let storage = storage_of(space);
                let i8t = self.b.int(8);
                let sp = self.b.ptr(storage, i8t);
                let gp = self.b.ptr(Storage::Generic, i8t);
                let w = self.b.width(ty);
                let it = self.b.int(w);
                let shared = space == StateSpace::Shared;
                if to {
                    let p = self.b.op(op::CONVERT_U_TO_PTR, gp, &[v]);
                    let q = self.b.op(op::GENERIC_CAST_TO_PTR, sp, &[p]);
                    if shared {
                        let off = self.shared_ptr_to_int(q);
                        let u64t = self.b.int(64);
                        self.coerce(off, u64t, it, false)
                    } else {
                        self.b.op(op::CONVERT_PTR_TO_U, it, &[q])
                    }
                } else {
                    let p = if shared {
                        let u64t = self.b.int(64);
                        let v64 = self.coerce(v, it, u64t, true);
                        self.shared_int_to_ptr(v64)
                    } else {
                        self.b.op(op::CONVERT_U_TO_PTR, sp, &[v])
                    };
                    let q = self.b.op(op::PTR_CAST_TO_GENERIC, gp, &[p]);
                    self.b.op(op::CONVERT_PTR_TO_U, it, &[q])
                }
            }
            _ => return Err("cvta needs a state space".into()),
        };
        self.write(arg(i, 0)?, r, ty, false)
    }

    fn mem_operands(&mut self, i: &Inst) -> Vec<u32> {
        let mut m = 0u32;
        let mut extra = vec![];
        if i.has("volatile") {
            m |= 1;
        }
        if let Ok(t) = inst_type(i) {
            let n = vec_width(i).max(1) as u32;
            m |= 2;
            extra.push(t.size() * n);
        }
        let mut v = vec![m];
        v.extend(extra);
        v
    }

    fn ld(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let n = vec_width(i);
        let space = space_of(i);
        let et = if t == Scalar::Pred { self.b.int(8) } else { self.ty_of(t) };
        let vt = if n > 0 { self.b.vector(et, n as u32) } else { et };
        let ptr = self.address(arg(i, 1)?, space, vt)?;
        let mem = self.mem_operands(i);
        let mut ops = vec![ptr];
        ops.extend(mem);
        let v = self.b.op(op::LOAD, vt, &ops);
        let dst = arg(i, 0)?;
        if n > 0 {
            let parts = match dst {
                Operand::Vector(p) => p.clone(),
                _ => return Err("vector load needs a vector destination".into()),
            };
            for (k, p) in parts.iter().enumerate() {
                let x = self.b.op(op::COMPOSITE_EXTRACT, et, &[v, k as u32]);
                self.write(p, x, et, t.is_signed())?;
            }
            Ok(())
        } else {
            self.write(dst, v, et, t.is_signed())
        }
    }

    fn st(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let n = vec_width(i);
        let space = space_of(i);
        let et = if t == Scalar::Pred { self.b.int(8) } else { self.ty_of(t) };
        let vt = if n > 0 { self.b.vector(et, n as u32) } else { et };
        let ptr = self.address(arg(i, 0)?, space, vt)?;
        let src = arg(i, 1)?;
        let v = if n > 0 {
            let parts = match src {
                Operand::Vector(p) => p.clone(),
                _ => return Err("vector store needs a vector source".into()),
            };
            let mut vals = Vec::new();
            for p in &parts {
                vals.push(self.read(p, et, t.is_signed())?);
            }
            self.b.op(op::COMPOSITE_CONSTRUCT, vt, &vals)
        } else {
            self.read(src, et, t.is_signed())?
        };
        let mut ops = vec![ptr, v];
        ops.extend(self.mem_operands(i));
        self.b.emit(op::STORE, &ops);
        Ok(())
    }

    // -------------------------------------------------------- comparisons

    fn compare(&mut self, cmp: &str, t: Scalar, a: Id, b: Id) -> R<Id> {
        let bt = self.b.bool();
        let k = t.kind();
        let opc = if k == ScalarKind::Float {
            match cmp {
                "eq" => op::F_ORD_EQUAL,
                "ne" => op::F_ORD_NOT_EQUAL,
                "lt" => op::F_ORD_LESS_THAN,
                "le" => op::F_ORD_LESS_THAN_EQUAL,
                "gt" => op::F_ORD_GREATER_THAN,
                "ge" => op::F_ORD_GREATER_THAN_EQUAL,
                "equ" => op::F_UNORD_EQUAL,
                "neu" => op::F_UNORD_NOT_EQUAL,
                "ltu" => op::F_UNORD_LESS_THAN,
                "leu" => op::F_UNORD_LESS_THAN_EQUAL,
                "gtu" => op::F_UNORD_GREATER_THAN,
                "geu" => op::F_UNORD_GREATER_THAN_EQUAL,
                "num" => op::ORDERED,
                "nan" => op::UNORDERED,
                _ => return Err(format!("unknown float comparison {cmp}")),
            }
        } else if k == ScalarKind::Pred {
            match cmp {
                "eq" => op::LOGICAL_EQUAL,
                "ne" => op::LOGICAL_NOT_EQUAL,
                _ => return Err(format!("unsupported predicate comparison {cmp}")),
            }
        } else {
            let s = k == ScalarKind::Signed;
            match cmp {
                "eq" => op::I_EQUAL,
                "ne" => op::I_NOT_EQUAL,
                "lt" => if s { op::S_LESS_THAN } else { op::U_LESS_THAN },
                "le" => if s { op::S_LESS_THAN_EQUAL } else { op::U_LESS_THAN_EQUAL },
                "gt" => if s { op::S_GREATER_THAN } else { op::U_GREATER_THAN },
                "ge" => if s { op::S_GREATER_THAN_EQUAL } else { op::U_GREATER_THAN_EQUAL },
                "lo" => op::U_LESS_THAN,
                "ls" => op::U_LESS_THAN_EQUAL,
                "hi" => op::U_GREATER_THAN,
                "hs" => op::U_GREATER_THAN_EQUAL,
                _ => return Err(format!("unknown integer comparison {cmp}")),
            }
        };
        Ok(self.b.op(opc, bt, &[a, b]))
    }

    fn setp(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let cmp = i.mods.first().ok_or("setp needs a comparison")?.clone();
        let boolop = i.mods.iter().find(|m| matches!(m.as_str(), "and" | "or" | "xor")).cloned();
        let ty = self.ty_of(t);
        let a = self.read(arg(i, 1)?, ty, t.is_signed())?;
        let b = self.read(arg(i, 2)?, ty, t.is_signed())?;
        let bt = self.b.bool();
        let mut r = self.compare(&cmp, t, a, b)?;
        let (p, q) = match arg(i, 0)? {
            Operand::Pair(p, q) => ((**p).clone(), Some((**q).clone())),
            p => (p.clone(), None),
        };
        let mut nr = self.b.op(op::LOGICAL_NOT, bt, &[r]);
        if let Some(bo) = boolop {
            let c = self.read(arg(i, 3)?, bt, false)?;
            let opc = match bo.as_str() {
                "and" => op::LOGICAL_AND,
                "or" => op::LOGICAL_OR,
                _ => op::LOGICAL_NOT_EQUAL,
            };
            r = self.b.op(opc, bt, &[r, c]);
            nr = self.b.op(opc, bt, &[nr, c]);
        }
        self.write(&p, r, bt, false)?;
        if let Some(q) = q {
            self.write(&q, nr, bt, false)?;
        }
        Ok(())
    }

    fn testp(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let bt = self.b.bool();
        let a = self.read(arg(i, 1)?, ty, false)?;
        let r = if i.has("finite") {
            self.b.op(op::IS_FINITE, bt, &[a])
        } else if i.has("infinite") {
            self.b.op(op::IS_INF, bt, &[a])
        } else if i.has("number") {
            self.b.op(op::ORDERED, bt, &[a, a])
        } else if i.has("notanumber") {
            self.b.op(op::IS_NAN, bt, &[a])
        } else if i.has("normal") {
            self.b.op(op::IS_NORMAL, bt, &[a])
        } else if i.has("subnormal") {
            let fin = self.b.op(op::IS_FINITE, bt, &[a]);
            let norm = self.b.op(op::IS_NORMAL, bt, &[a]);
            let nn = self.b.op(op::LOGICAL_NOT, bt, &[norm]);
            let z = self.b.const_bits(ty, 0);
            let nz = self.b.op(op::F_UNORD_NOT_EQUAL, bt, &[a, z]);
            let x = self.b.op(op::LOGICAL_AND, bt, &[fin, nn]);
            self.b.op(op::LOGICAL_AND, bt, &[x, nz])
        } else {
            return Err("unknown testp".into());
        };
        self.write(arg(i, 0)?, r, bt, false)
    }

    // -------------------------------------------------------- arithmetic

    fn sat01(&mut self, ty: Id, v: Id) -> Id {
        let z = self.b.const_bits(ty, 0);
        let one = self.imm_int(1, ty);
        let a = self.b.ext(ty, cl::FMAX, &[v, z]);
        self.b.ext(ty, cl::FMIN, &[a, one])
    }

    fn carry_var(&mut self) -> Id {
        if let Some(c) = self.f.carry {
            return c;
        }
        let u32t = self.b.int(32);
        let v = self.local(u32t);
        self.f.carry = Some(v);
        v
    }

    fn binary(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        if is_packed_half(t) {
            return self.packed_binary(i, t);
        }
        let ty = self.ty_of(t);
        let k = t.kind();
        let signed = k == ScalarKind::Signed;
        let dst = arg(i, 0)?;
        // wide multiply: inputs are half the destination width
        if (i.op == "mul") && i.has("wide") {
            let it = self.ty_of(t);
            let w = self.b.width(it);
            let wt = self.b.int(w * 2);
            let a = self.read(arg(i, 1)?, it, signed)?;
            let b = self.read(arg(i, 2)?, it, signed)?;
            let cv = if signed { op::S_CONVERT } else { op::U_CONVERT };
            let a = self.b.op(cv, wt, &[a]);
            let b = self.b.op(cv, wt, &[b]);
            let r = self.b.op(op::I_MUL, wt, &[a, b]);
            return self.write(dst, r, wt, signed);
        }
        let shift = i.op == "shl" || i.op == "shr";
        let a = self.read(arg(i, 1)?, ty, signed)?;
        let b = if shift {
            let u32t = self.b.int(32);
            self.read(arg(i, 2)?, u32t, false)?
        } else {
            self.read(arg(i, 2)?, ty, signed)?
        };
        let float = k == ScalarKind::Float;
        let pred = k == ScalarKind::Pred;
        let r = match (i.op.as_str(), float) {
            ("add", true) => self.b.op(op::F_ADD, ty, &[a, b]),
            ("sub", true) => self.b.op(op::F_SUB, ty, &[a, b]),
            ("mul", true) => self.b.op(op::F_MUL, ty, &[a, b]),
            ("div", true) => {
                if i.has("approx") && t == Scalar::F32 {
                    self.b.ext(ty, cl::NATIVE_DIVIDE, &[a, b])
                } else {
                    self.b.op(op::F_DIV, ty, &[a, b])
                }
            }
            ("rem", true) => self.b.op(op::F_REM, ty, &[a, b]),
            ("min", true) => self.b.ext(ty, cl::FMIN, &[a, b]),
            ("max", true) => self.b.ext(ty, cl::FMAX, &[a, b]),
            ("copysign", true) => self.b.ext(ty, cl::COPYSIGN, &[b, a]),
            ("add", false) | ("addc", false) | ("sub", false) | ("subc", false) => {
                let is_add = i.op.starts_with("add");
                let use_cin = i.op == "addc" || i.op == "subc";
                let set_cout = i.has("cc");
                if !use_cin && !set_cout {
                    if i.has("sat") && t == Scalar::S32 {
                        self.b.ext(ty, if is_add { cl::S_ADD_SAT } else { cl::S_SUB_SAT }, &[a, b])
                    } else {
                        self.b.op(if is_add { op::I_ADD } else { op::I_SUB }, ty, &[a, b])
                    }
                } else {
                    self.add_carry(ty, a, b, is_add, use_cin, set_cout)
                }
            }
            ("mul", false) => {
                if i.has("hi") {
                    self.b.ext(ty, if signed { cl::S_MUL_HI } else { cl::U_MUL_HI }, &[a, b])
                } else {
                    self.b.op(op::I_MUL, ty, &[a, b])
                }
            }
            ("mul24", false) => {
                if i.has("hi") {
                    return Err("mul24.hi is not supported".into());
                }
                self.b.ext(ty, if signed { cl::S_MUL24 } else { cl::U_MUL24 }, &[a, b])
            }
            ("div", false) => self.b.op(if signed { op::S_DIV } else { op::U_DIV }, ty, &[a, b]),
            ("rem", false) => self.b.op(if signed { op::S_REM } else { op::U_MOD }, ty, &[a, b]),
            ("min", false) => self.b.ext(ty, if signed { cl::S_MIN } else { cl::U_MIN }, &[a, b]),
            ("max", false) => self.b.ext(ty, if signed { cl::S_MAX } else { cl::U_MAX }, &[a, b]),
            ("and", false) => self.b.op(if pred { op::LOGICAL_AND } else { op::BITWISE_AND }, ty, &[a, b]),
            ("or", false) => self.b.op(if pred { op::LOGICAL_OR } else { op::BITWISE_OR }, ty, &[a, b]),
            ("xor", false) => self.b.op(if pred { op::LOGICAL_NOT_EQUAL } else { op::BITWISE_XOR }, ty, &[a, b]),
            ("shl", false) | ("shr", false) => {
                // PTX clamps shift amounts >= width
                let w = self.b.width(ty);
                let u32t = self.b.int(32);
                let wm1 = self.b.const_u32(w - 1);
                let amt = self.b.ext(u32t, cl::U_MIN, &[b, wm1]);
                let amt_t = self.coerce(amt, u32t, ty, false);
                let arith = i.op == "shr" && signed;
                let opc = if i.op == "shl" {
                    op::SHIFT_LEFT_LOGICAL
                } else if arith {
                    op::SHIFT_RIGHT_ARITHMETIC
                } else {
                    op::SHIFT_RIGHT_LOGICAL
                };
                let r = self.b.op(opc, ty, &[a, amt_t]);
                if arith {
                    r
                } else {
                    let bt = self.b.bool();
                    let big = self.b.op(op::U_GREATER_THAN, bt, &[b, wm1]);
                    let z = self.b.const_bits(ty, 0);
                    self.b.op(op::SELECT, ty, &[big, z, r])
                }
            }
            _ => return Err(format!("unsupported {} for {:?}", i.op, t)),
        };
        let r = if float && i.has("sat") { self.sat01(ty, r) } else { r };
        self.write(dst, r, ty, signed)
    }

    /// Add/sub with optional carry-in and carry-out through the carry flag.
    fn add_carry(&mut self, ty: Id, a: Id, b: Id, add: bool, cin: bool, cout: bool) -> Id {
        let u32t = self.b.int(32);
        let pair = self.b.ty(TypeKey::Struct(vec![ty, ty]));
        let cv = self.carry_var();
        let opc = if add { op::I_ADD_CARRY } else { op::I_SUB_BORROW };
        let s1 = self.b.op(opc, pair, &[a, b]);
        let mut r = self.b.op(op::COMPOSITE_EXTRACT, ty, &[s1, 0]);
        let mut c = self.b.op(op::COMPOSITE_EXTRACT, ty, &[s1, 1]);
        if cin {
            let cf = self.b.op(op::LOAD, u32t, &[cv]);
            let cft = self.coerce(cf, u32t, ty, false);
            let s2 = self.b.op(opc, pair, &[r, cft]);
            r = self.b.op(op::COMPOSITE_EXTRACT, ty, &[s2, 0]);
            let c2 = self.b.op(op::COMPOSITE_EXTRACT, ty, &[s2, 1]);
            c = self.b.op(op::BITWISE_OR, ty, &[c, c2]);
        }
        if cout {
            let c32 = self.coerce(c, ty, u32t, false);
            self.b.emit(op::STORE, &[cv, c32]);
        }
        r
    }

    fn mad(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        if is_packed_half(t) {
            let f32x2 = self.f32x2_ty();
            let a = self.read_packed_f32x2(arg(i, 1)?, t)?;
            let b = self.read_packed_f32x2(arg(i, 2)?, t)?;
            let c = self.read_packed_f32x2(arg(i, 3)?, t)?;
            let r = self.b.ext(f32x2, cl::FMA, &[a, b, c]);
            return self.write_packed(arg(i, 0)?, r, t);
        }
        let ty = self.ty_of(t);
        let signed = t.is_signed();
        let dst = arg(i, 0)?;
        if t.is_float() {
            let a = self.read(arg(i, 1)?, ty, false)?;
            let b = self.read(arg(i, 2)?, ty, false)?;
            let c = self.read(arg(i, 3)?, ty, false)?;
            let mut r = self.b.ext(ty, cl::FMA, &[a, b, c]);
            if i.has("sat") {
                r = self.sat01(ty, r);
            }
            return self.write(dst, r, ty, false);
        }
        if i.has("wide") {
            let w = self.b.width(ty);
            let wt = self.b.int(w * 2);
            let a = self.read(arg(i, 1)?, ty, signed)?;
            let b = self.read(arg(i, 2)?, ty, signed)?;
            let c = self.read(arg(i, 3)?, wt, signed)?;
            let cv = if signed { op::S_CONVERT } else { op::U_CONVERT };
            let a = self.b.op(cv, wt, &[a]);
            let b = self.b.op(cv, wt, &[b]);
            let m = self.b.op(op::I_MUL, wt, &[a, b]);
            let r = self.b.op(op::I_ADD, wt, &[m, c]);
            return self.write(dst, r, wt, signed);
        }
        let a = self.read(arg(i, 1)?, ty, signed)?;
        let b = self.read(arg(i, 2)?, ty, signed)?;
        let c = self.read(arg(i, 3)?, ty, signed)?;
        let m = if i.has("hi") {
            self.b.ext(ty, if signed { cl::S_MUL_HI } else { cl::U_MUL_HI }, &[a, b])
        } else if i.op == "mad24" {
            self.b.ext(ty, if signed { cl::S_MUL24 } else { cl::U_MUL24 }, &[a, b])
        } else {
            self.b.op(op::I_MUL, ty, &[a, b])
        };
        let r = if i.has("cc") || i.op == "madc" {
            self.add_carry(ty, m, c, true, i.op == "madc", i.has("cc"))
        } else {
            self.b.op(op::I_ADD, ty, &[m, c])
        };
        self.write(dst, r, ty, signed)
    }

    fn unary(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        if is_packed_half(t) {
            let f32x2 = self.f32x2_ty();
            let a = self.read_packed_f32x2(arg(i, 1)?, t)?;
            let r = match i.op.as_str() {
                "neg" => self.b.op(op::F_NEGATE, f32x2, &[a]),
                "abs" => self.b.ext(f32x2, cl::FABS, &[a]),
                o => return Err(format!("packed-half {o} is not supported")),
            };
            return self.write_packed(arg(i, 0)?, r, t);
        }
        let ty = self.ty_of(t);
        let float = t.is_float();
        let signed = t.is_signed();
        let a = self.read(arg(i, 1)?, ty, signed)?;
        let approx = i.has("approx");
        let u32t = self.b.int(32);
        let (r, rt) = match i.op.as_str() {
            "abs" if float => (self.b.ext(ty, cl::FABS, &[a]), ty),
            "abs" => (self.b.ext(ty, cl::S_ABS, &[a]), ty),
            "neg" if float => (self.b.op(op::F_NEGATE, ty, &[a]), ty),
            "neg" => (self.b.op(op::S_NEGATE, ty, &[a]), ty),
            "not" if t == Scalar::Pred => (self.b.op(op::LOGICAL_NOT, ty, &[a]), ty),
            "not" => (self.b.op(op::NOT, ty, &[a]), ty),
            "cnot" => {
                let bt = self.b.bool();
                let z = self.b.const_bits(ty, 0);
                let one = self.b.const_bits(ty, 1);
                let c = self.b.op(op::I_EQUAL, bt, &[a, z]);
                (self.b.op(op::SELECT, ty, &[c, one, z]), ty)
            }
            "popc" => {
                let r = self.b.ext(ty, cl::POPCOUNT, &[a]);
                (self.coerce(r, ty, u32t, false), u32t)
            }
            "clz" => {
                let r = self.b.ext(ty, cl::CLZ, &[a]);
                (self.coerce(r, ty, u32t, false), u32t)
            }
            "brev" => (self.bit_reverse(ty, a), ty),
            "rcp" => {
                if approx && t == Scalar::F32 {
                    (self.b.ext(ty, cl::NATIVE_RECIP, &[a]), ty)
                } else {
                    let one = self.imm_int(1, ty);
                    (self.b.op(op::F_DIV, ty, &[one, a]), ty)
                }
            }
            "sqrt" => (self.b.ext(ty, if approx && t == Scalar::F32 { cl::NATIVE_SQRT } else { cl::SQRT }, &[a]), ty),
            "rsqrt" => (self.b.ext(ty, if t == Scalar::F32 { cl::NATIVE_RSQRT } else { cl::RSQRT }, &[a]), ty),
            "sin" => (self.b.ext(ty, cl::NATIVE_SIN, &[a]), ty),
            "cos" => (self.b.ext(ty, cl::NATIVE_COS, &[a]), ty),
            "lg2" => (self.b.ext(ty, cl::NATIVE_LOG2, &[a]), ty),
            "ex2" => (self.b.ext(ty, cl::NATIVE_EXP2, &[a]), ty),
            "tanh" => (self.b.ext(ty, cl::TANH, &[a]), ty),
            o => return Err(format!("unsupported {o}")),
        };
        let r = if float && i.has("sat") { self.sat01(ty, r) } else { r };
        self.write(arg(i, 0)?, r, rt, signed)
    }

    fn bit_reverse(&mut self, ty: Id, v: Id) -> Id {
        let w = self.b.width(ty);
        let masks: [(u32, u64); 6] = [
            (1, 0x5555_5555_5555_5555),
            (2, 0x3333_3333_3333_3333),
            (4, 0x0f0f_0f0f_0f0f_0f0f),
            (8, 0x00ff_00ff_00ff_00ff),
            (16, 0x0000_ffff_0000_ffff),
            (32, 0x0000_0000_ffff_ffff),
        ];
        let mut x = v;
        for (s, m) in masks {
            if s >= w {
                break;
            }
            let mc = self.b.const_bits(ty, m);
            let sc = self.b.const_bits(ty, s as u64);
            let lo = self.b.op(op::BITWISE_AND, ty, &[x, mc]);
            let lo = self.b.op(op::SHIFT_LEFT_LOGICAL, ty, &[lo, sc]);
            let hi = self.b.op(op::SHIFT_RIGHT_LOGICAL, ty, &[x, sc]);
            let hi = self.b.op(op::BITWISE_AND, ty, &[hi, mc]);
            x = self.b.op(op::BITWISE_OR, ty, &[lo, hi]);
        }
        x
    }

    fn bfe(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let signed = t.is_signed();
        let w = self.b.width(ty);
        let u32t = self.b.int(32);
        let bt = self.b.bool();
        let a = self.read(arg(i, 1)?, ty, signed)?;
        let pos = self.read(arg(i, 2)?, u32t, false)?;
        let len = self.read(arg(i, 3)?, u32t, false)?;
        let ff = self.b.const_u32(0xff);
        let pos = self.b.op(op::BITWISE_AND, u32t, &[pos, ff]);
        let len = self.b.op(op::BITWISE_AND, u32t, &[len, ff]);
        let wc = self.b.const_u32(w);
        let wm1 = self.b.const_u32(w - 1);
        let zero32 = self.b.const_u32(0);
        let zero = self.b.const_bits(ty, 0);
        let shr = if signed { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL };
        // fits: pos + len <= w  -> (a << (w-pos-len)) >> (w-len)
        let end = self.b.op(op::I_ADD, u32t, &[pos, len]);
        let fits = self.b.op(op::U_LESS_THAN_EQUAL, bt, &[end, wc]);
        let l1 = self.b.op(op::I_SUB, u32t, &[wc, end]);
        let l1 = self.b.op(op::SELECT, u32t, &[fits, l1, zero32]);
        let r1 = self.b.op(op::I_SUB, u32t, &[wc, len]);
        let r1 = self.b.ext(u32t, cl::U_MIN, &[r1, wm1]);
        let l1t = self.coerce(l1, u32t, ty, false);
        let r1t = self.coerce(r1, u32t, ty, false);
        let x = self.b.op(op::SHIFT_LEFT_LOGICAL, ty, &[a, l1t]);
        let v_fit = self.b.op(shr, ty, &[x, r1t]);
        // !fits: a >> min(pos, w-1), or sign/zero fill when pos >= w
        let pc = self.b.ext(u32t, cl::U_MIN, &[pos, wm1]);
        let pct = self.coerce(pc, u32t, ty, false);
        let v_over = self.b.op(shr, ty, &[a, pct]);
        let pos_ge = self.b.op(op::U_GREATER_THAN_EQUAL, bt, &[pos, wc]);
        let v_over = if signed { v_over } else { self.b.op(op::SELECT, ty, &[pos_ge, zero, v_over]) };
        let r = self.b.op(op::SELECT, ty, &[fits, v_fit, v_over]);
        let len0 = self.b.op(op::I_EQUAL, bt, &[len, zero32]);
        let r = self.b.op(op::SELECT, ty, &[len0, zero, r]);
        self.write(arg(i, 0)?, r, ty, signed)
    }

    fn bfi(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let w = self.b.width(ty);
        let u32t = self.b.int(32);
        let bt = self.b.bool();
        let a = self.read(arg(i, 1)?, ty, false)?;
        let b = self.read(arg(i, 2)?, ty, false)?;
        let pos = self.read(arg(i, 3)?, u32t, false)?;
        let len = self.read(arg(i, 4)?, u32t, false)?;
        let ff = self.b.const_u32(0xff);
        let pos = self.b.op(op::BITWISE_AND, u32t, &[pos, ff]);
        let len = self.b.op(op::BITWISE_AND, u32t, &[len, ff]);
        let wc = self.b.const_u32(w);
        let wm1 = self.b.const_u32(w - 1);
        let ones = self.b.const_bits(ty, u64::MAX);
        let one = self.b.const_bits(ty, 1);
        let lc = self.b.ext(u32t, cl::U_MIN, &[len, wm1]);
        let lct = self.coerce(lc, u32t, ty, false);
        let m = self.b.op(op::SHIFT_LEFT_LOGICAL, ty, &[one, lct]);
        let m = self.b.op(op::I_SUB, ty, &[m, one]);
        let full = self.b.op(op::U_GREATER_THAN_EQUAL, bt, &[len, wc]);
        let m = self.b.op(op::SELECT, ty, &[full, ones, m]);
        let pc = self.b.ext(u32t, cl::U_MIN, &[pos, wm1]);
        let pct = self.coerce(pc, u32t, ty, false);
        let m = self.b.op(op::SHIFT_LEFT_LOGICAL, ty, &[m, pct]);
        let ins = self.b.op(op::SHIFT_LEFT_LOGICAL, ty, &[a, pct]);
        let ins = self.b.op(op::BITWISE_AND, ty, &[ins, m]);
        let nm = self.b.op(op::NOT, ty, &[m]);
        let keep = self.b.op(op::BITWISE_AND, ty, &[b, nm]);
        let r = self.b.op(op::BITWISE_OR, ty, &[keep, ins]);
        let pos_ge = self.b.op(op::U_GREATER_THAN_EQUAL, bt, &[pos, wc]);
        let r = self.b.op(op::SELECT, ty, &[pos_ge, b, r]);
        self.write(arg(i, 0)?, r, ty, false)
    }

    /// prmt.b32 (default mode only).
    fn prmt(&mut self, i: &Inst) -> R<()> {
        if i.mods.iter().any(|m| matches!(m.as_str(), "f4e" | "b4e" | "rc8" | "ecl" | "ecr" | "rc16")) {
            return Err("prmt modes other than default are not supported yet".into());
        }
        let u32t = self.b.int(32);
        let u64t = self.b.int(64);
        let bt = self.b.bool();
        let a = self.read(arg(i, 1)?, u32t, false)?;
        let b = self.read(arg(i, 2)?, u32t, false)?;
        let c = self.read(arg(i, 3)?, u32t, false)?;
        let a64 = self.b.op(op::U_CONVERT, u64t, &[a]);
        let b64 = self.b.op(op::U_CONVERT, u64t, &[b]);
        let c32 = self.b.const_u64(32);
        let hi = self.b.op(op::SHIFT_LEFT_LOGICAL, u64t, &[b64, c32]);
        let pool = self.b.op(op::BITWISE_OR, u64t, &[hi, a64]);
        let mut r = self.b.const_u32(0);
        for k in 0..4u32 {
            let sh = self.b.const_u32(4 * k);
            let sel = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[c, sh]);
            let seven = self.b.const_u32(7);
            let eight = self.b.const_u32(8);
            let idx = self.b.op(op::BITWISE_AND, u32t, &[sel, seven]);
            let three = self.b.const_u32(3);
            let bits = self.b.op(op::SHIFT_LEFT_LOGICAL, u32t, &[idx, three]);
            let bits64 = self.b.op(op::U_CONVERT, u64t, &[bits]);
            let byte = self.b.op(op::SHIFT_RIGHT_LOGICAL, u64t, &[pool, bits64]);
            let byte = self.b.op(op::U_CONVERT, u32t, &[byte]);
            let ff = self.b.const_u32(0xff);
            let byte = self.b.op(op::BITWISE_AND, u32t, &[byte, ff]);
            // sign replicate mode
            let msb = self.b.op(op::BITWISE_AND, u32t, &[sel, eight]);
            let zero = self.b.const_u32(0);
            let rep = self.b.op(op::I_NOT_EQUAL, bt, &[msb, zero]);
            let x80 = self.b.const_u32(0x80);
            let sb = self.b.op(op::BITWISE_AND, u32t, &[byte, x80]);
            let neg = self.b.op(op::I_NOT_EQUAL, bt, &[sb, zero]);
            let fill = self.b.op(op::SELECT, u32t, &[neg, ff, zero]);
            let byte = self.b.op(op::SELECT, u32t, &[rep, fill, byte]);
            let pos = self.b.const_u32(8 * k);
            let byte = self.b.op(op::SHIFT_LEFT_LOGICAL, u32t, &[byte, pos]);
            r = self.b.op(op::BITWISE_OR, u32t, &[r, byte]);
        }
        self.write(arg(i, 0)?, r, u32t, false)
    }

    /// shf.{l,r}.{wrap,clamp}.b32 d, a, b, c — funnel shift of b:a.
    fn shf(&mut self, i: &Inst) -> R<()> {
        let u32t = self.b.int(32);
        let u64t = self.b.int(64);
        let a = self.read(arg(i, 1)?, u32t, false)?;
        let b = self.read(arg(i, 2)?, u32t, false)?;
        let c = self.read(arg(i, 3)?, u32t, false)?;
        let n = if i.has("clamp") {
            let c32 = self.b.const_u32(32);
            self.b.ext(u32t, cl::U_MIN, &[c, c32])
        } else {
            let m = self.b.const_u32(31);
            self.b.op(op::BITWISE_AND, u32t, &[c, m])
        };
        let a64 = self.b.op(op::U_CONVERT, u64t, &[a]);
        let b64 = self.b.op(op::U_CONVERT, u64t, &[b]);
        let s32 = self.b.const_u64(32);
        let hi = self.b.op(op::SHIFT_LEFT_LOGICAL, u64t, &[b64, s32]);
        let x = self.b.op(op::BITWISE_OR, u64t, &[hi, a64]);
        let n64 = self.b.op(op::U_CONVERT, u64t, &[n]);
        let r = if i.has("l") {
            let y = self.b.op(op::SHIFT_LEFT_LOGICAL, u64t, &[x, n64]);
            let y = self.b.op(op::SHIFT_RIGHT_LOGICAL, u64t, &[y, s32]);
            self.b.op(op::U_CONVERT, u32t, &[y])
        } else {
            let y = self.b.op(op::SHIFT_RIGHT_LOGICAL, u64t, &[x, n64]);
            self.b.op(op::U_CONVERT, u32t, &[y])
        };
        self.write(arg(i, 0)?, r, u32t, false)
    }

    // -------------------------------------------------------- conversions

    fn round_int(&mut self, i: &Inst, ty: Id, v: Id, default_trunc: bool) -> Id {
        if i.has("rni") {
            self.b.ext(ty, cl::RINT, &[v])
        } else if i.has("rzi") {
            self.b.ext(ty, cl::TRUNC, &[v])
        } else if i.has("rmi") {
            self.b.ext(ty, cl::FLOOR, &[v])
        } else if i.has("rpi") {
            self.b.ext(ty, cl::CEIL, &[v])
        } else if default_trunc {
            self.b.ext(ty, cl::TRUNC, &[v])
        } else {
            v
        }
    }

    fn rounding_deco(&mut self, i: &Inst, id: Id) {
        let m = if i.has("rz") {
            Some(1)
        } else if i.has("rp") {
            Some(2)
        } else if i.has("rm") {
            Some(3)
        } else {
            None
        };
        if let Some(m) = m {
            self.b.decorate(id, deco::FP_ROUNDING_MODE, &[m]);
        }
    }

    // ------------------------------------------------------ packed half

    /// `vector(f32, 2)` — the internal representation for packed-half
    /// arithmetic. Computing in f32 and rounding once at the end is at least
    /// as accurate as per-lane f16 rounding, which is fine for inference.
    fn f32x2_ty(&mut self) -> Id {
        let f = self.b.float(32);
        self.b.vector(f, 2)
    }

    /// Read a packed-half operand (a 32-bit register) as `vector(f32, 2)`,
    /// lane 0 = low 16 bits, lane 1 = high 16 bits.
    fn read_packed_f32x2(&mut self, o: &Operand, t: Scalar) -> R<Id> {
        let u32t = self.b.int(32);
        let bits = self.read(o, u32t, false)?;
        let f32x2 = self.f32x2_ty();
        match t {
            Scalar::F16x2 => {
                let half = self.b.float(16);
                let h2 = self.b.vector(half, 2);
                let hv = self.b.op(op::BITCAST, h2, &[bits]);
                Ok(self.b.op(op::F_CONVERT, f32x2, &[hv]))
            }
            Scalar::BF16x2 => {
                let u16t = self.b.int(16);
                let i16x2 = self.b.vector(u16t, 2);
                let iv = self.b.op(op::BITCAST, i16x2, &[bits]);
                let f32t = self.b.float(32);
                let s16 = self.b.const_u32(16);
                let mut lanes = Vec::new();
                for k in 0..2u32 {
                    // bf16 -> f32: place the 16 bits in the high half of an f32
                    let e = self.b.op(op::COMPOSITE_EXTRACT, u16t, &[iv, k]);
                    let w = self.b.op(op::U_CONVERT, u32t, &[e]);
                    let w = self.b.op(op::SHIFT_LEFT_LOGICAL, u32t, &[w, s16]);
                    lanes.push(self.b.op(op::BITCAST, f32t, &[w]));
                }
                Ok(self.b.op(op::COMPOSITE_CONSTRUCT, f32x2, &lanes))
            }
            _ => Err("read_packed_f32x2 on a non-packed type".into()),
        }
    }

    /// Round an f32 to bf16, returning the 16-bit pattern (round-to-nearest
    /// even, quiet NaNs preserved). Mirrors the scalar path in `cvt_bf16`.
    fn f32_to_bf16_bits(&mut self, f: Id) -> Id {
        let u32t = self.b.int(32);
        let u16t = self.b.int(16);
        let s16 = self.b.const_u32(16);
        let x = self.b.op(op::BITCAST, u32t, &[f]);
        let one = self.b.const_u32(1);
        let lsb = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[x, s16]);
        let lsb = self.b.op(op::BITWISE_AND, u32t, &[lsb, one]);
        let bias = self.b.const_u32(0x7fff);
        let bias = self.b.op(op::I_ADD, u32t, &[bias, lsb]);
        let rounded = self.b.op(op::I_ADD, u32t, &[x, bias]);
        let bt = self.b.bool();
        let nan = self.b.op(op::IS_NAN, bt, &[f]);
        let qnan = self.b.const_u32(0x7fc0_0000);
        let rounded = self.b.op(op::SELECT, u32t, &[nan, qnan, rounded]);
        let h = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[rounded, s16]);
        self.b.op(op::U_CONVERT, u16t, &[h])
    }

    /// Pack a `vector(f32, 2)` back into the 32-bit register bits for `t`.
    fn pack_from_f32x2(&mut self, v: Id, t: Scalar) -> Id {
        let u32t = self.b.int(32);
        match t {
            Scalar::F16x2 => {
                let half = self.b.float(16);
                let h2 = self.b.vector(half, 2);
                let hv = self.b.op(op::F_CONVERT, h2, &[v]);
                self.b.op(op::BITCAST, u32t, &[hv])
            }
            Scalar::BF16x2 => {
                let f32t = self.b.float(32);
                let u16t = self.b.int(16);
                let i16x2 = self.b.vector(u16t, 2);
                let mut lanes = Vec::new();
                for k in 0..2u32 {
                    let f = self.b.op(op::COMPOSITE_EXTRACT, f32t, &[v, k]);
                    lanes.push(self.f32_to_bf16_bits(f));
                }
                let iv = self.b.op(op::COMPOSITE_CONSTRUCT, i16x2, &lanes);
                self.b.op(op::BITCAST, u32t, &[iv])
            }
            _ => v,
        }
    }

    /// Write a `vector(f32, 2)` result to a packed-half destination register.
    fn write_packed(&mut self, d: &Operand, v: Id, t: Scalar) -> R<()> {
        let bits = self.pack_from_f32x2(v, t);
        let u32t = self.b.int(32);
        self.write(d, bits, u32t, false)
    }

    /// `add/sub/mul/min/max.{f16x2,bf16x2}`.
    fn packed_binary(&mut self, i: &Inst, t: Scalar) -> R<()> {
        let f32x2 = self.f32x2_ty();
        let a = self.read_packed_f32x2(arg(i, 1)?, t)?;
        let b = self.read_packed_f32x2(arg(i, 2)?, t)?;
        let r = match i.op.as_str() {
            "add" => self.b.op(op::F_ADD, f32x2, &[a, b]),
            "sub" => self.b.op(op::F_SUB, f32x2, &[a, b]),
            "mul" => self.b.op(op::F_MUL, f32x2, &[a, b]),
            "min" => self.b.ext(f32x2, cl::FMIN, &[a, b]),
            "max" => self.b.ext(f32x2, cl::FMAX, &[a, b]),
            o => return Err(format!("packed-half {o} is not supported")),
        };
        self.write_packed(arg(i, 0)?, r, t)
    }

    fn cvt(&mut self, i: &Inst) -> R<()> {
        let (dt, at) = two_types(i)?;
        let dty = self.ty_of(dt);
        let aty = self.ty_of(at);
        let dst = arg(i, 0)?;
        let src = arg(i, 1)?;
        if is_packed_half(dt) {
            // pack two scalars: `cvt.rn.f16x2.f32 d, hi, lo`
            //   d[31:16] = cvt(hi)   (lane 1)
            //   d[15:0]  = cvt(lo)   (lane 0)
            let hi_op = arg(i, 1)?;
            let lo_op = arg(i, 2).map_err(|_| "packed-half cvt needs two source operands".to_string())?;
            let f32t = self.b.float(32);
            let to_f32 = |me: &mut Self, o: &Operand| -> R<Id> {
                let v = me.read(o, aty, at.is_signed())?;
                Ok(if at == Scalar::F32 {
                    v
                } else if at.is_float() {
                    me.b.op(op::F_CONVERT, f32t, &[v])
                } else if at.is_signed() {
                    me.b.op(op::CONVERT_S_TO_F, f32t, &[v])
                } else {
                    me.b.op(op::CONVERT_U_TO_F, f32t, &[v])
                })
            };
            let hi = to_f32(self, hi_op)?;
            let lo = to_f32(self, lo_op)?;
            let f32x2 = self.f32x2_ty();
            let v = self.b.op(op::COMPOSITE_CONSTRUCT, f32x2, &[lo, hi]);
            return self.write_packed(dst, v, dt);
        }
        if is_packed_half(at) {
            return Err("cvt from a packed-half type is not supported (unpack with mov.b32)".into());
        }
        let a = self.read(src, aty, at.is_signed())?;
        let (dk, ak) = (dt.kind(), at.kind());
        let float = |k| k == ScalarKind::Float;
        // bf16 is stored as 16-bit integers
        if dt == Scalar::BF16 || at == Scalar::BF16 {
            return self.cvt_bf16(i, dt, at, a, dst);
        }
        let r = if float(dk) && float(ak) {
            if dt == at {
                let r = self.round_int(i, dty, a, false);
                if i.has("sat") {
                    self.sat01(dty, r)
                } else {
                    r
                }
            } else {
                let r = self.b.op(op::F_CONVERT, dty, &[a]);
                self.rounding_deco(i, r);
                if i.has("sat") {
                    self.sat01(dty, r)
                } else {
                    r
                }
            }
        } else if dk != ScalarKind::Float && float(ak) {
            let rv = self.round_int(i, aty, a, true);
            let r = self.b.op(if dt.is_signed() { op::CONVERT_F_TO_S } else { op::CONVERT_F_TO_U }, dty, &[rv]);
            self.b.decorate(r, deco::SATURATED_CONVERSION, &[]);
            r
        } else if float(dk) {
            let r = self.b.op(if at.is_signed() { op::CONVERT_S_TO_F } else { op::CONVERT_U_TO_F }, dty, &[a]);
            self.rounding_deco(i, r);
            r
        } else {
            // integer <-> integer
            let r = self.coerce(a, aty, dty, at.is_signed());
            if i.has("sat") {
                self.int_saturate(dt, at, a, aty, r)?
            } else {
                r
            }
        };
        self.write(dst, r, dty, dt.is_signed())
    }

    /// Clamp integer `a` (type `at`) into the range of `dt`; `r` is the
    /// unsaturated conversion.
    fn int_saturate(&mut self, dt: Scalar, at: Scalar, a: Id, aty: Id, r: Id) -> R<Id> {
        let dw = dt.size() * 8;
        let aw = at.size() * 8;
        let (dmin, dmax): (i128, i128) = if dt.is_signed() {
            (-(1i128 << (dw - 1)), (1i128 << (dw - 1)) - 1)
        } else {
            (0, (1i128 << dw) - 1)
        };
        let (amin, amax): (i128, i128) = if at.is_signed() {
            (-(1i128 << (aw - 1)), (1i128 << (aw - 1)) - 1)
        } else {
            (0, (1i128 << aw) - 1)
        };
        let lo = dmin.max(amin);
        let hi = dmax.min(amax);
        let mut v = a;
        let (mn, mx) = if at.is_signed() { (cl::S_MIN, cl::S_MAX) } else { (cl::U_MIN, cl::U_MAX) };
        if hi < amax {
            let c = self.b.const_bits(aty, hi as u64);
            v = self.b.ext(aty, mn, &[v, c]);
        }
        if lo > amin {
            let c = self.b.const_bits(aty, lo as u64);
            v = self.b.ext(aty, mx, &[v, c]);
        }
        let dty = self.ty_of(dt);
        let _ = r;
        Ok(self.coerce(v, aty, dty, at.is_signed()))
    }

    fn cvt_bf16(&mut self, i: &Inst, dt: Scalar, at: Scalar, a: Id, dst: &Operand) -> R<()> {
        let u32t = self.b.int(32);
        let u16t = self.b.int(16);
        let f32t = self.b.float(32);
        let s16 = self.b.const_u32(16);
        if at == Scalar::BF16 {
            // widen to f32 by placing the bits in the upper half
            let x = self.b.op(op::U_CONVERT, u32t, &[a]);
            let x = self.b.op(op::SHIFT_LEFT_LOGICAL, u32t, &[x, s16]);
            let f = self.b.op(op::BITCAST, f32t, &[x]);
            if dt == Scalar::BF16 {
                return self.write(dst, a, u16t, false);
            }
            let dty = self.ty_of(dt);
            let r = if dt.is_float() {
                if dt == Scalar::F32 {
                    f
                } else {
                    self.b.op(op::F_CONVERT, dty, &[f])
                }
            } else {
                let rv = self.round_int(i, f32t, f, true);
                let r = self.b.op(if dt.is_signed() { op::CONVERT_F_TO_S } else { op::CONVERT_F_TO_U }, dty, &[rv]);
                self.b.decorate(r, deco::SATURATED_CONVERSION, &[]);
                r
            };
            return self.write(dst, r, dty, dt.is_signed());
        }
        // to bf16: convert to f32 first, then round-to-nearest-even
        let f = if at == Scalar::F32 {
            a
        } else if at.is_float() {
            self.b.op(op::F_CONVERT, f32t, &[a])
        } else {
            self.b.op(if at.is_signed() { op::CONVERT_S_TO_F } else { op::CONVERT_U_TO_F }, f32t, &[a])
        };
        let x = self.b.op(op::BITCAST, u32t, &[f]);
        let one = self.b.const_u32(1);
        let lsb = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[x, s16]);
        let lsb = self.b.op(op::BITWISE_AND, u32t, &[lsb, one]);
        let bias = self.b.const_u32(0x7fff);
        let bias = self.b.op(op::I_ADD, u32t, &[bias, lsb]);
        let rounded = if i.has("rz") { x } else { self.b.op(op::I_ADD, u32t, &[x, bias]) };
        // keep NaNs quiet NaNs
        let bt = self.b.bool();
        let nan = self.b.op(op::IS_NAN, bt, &[f]);
        let qnan = self.b.const_u32(0x7fc0_0000);
        let rounded = self.b.op(op::SELECT, u32t, &[nan, qnan, rounded]);
        let h = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[rounded, s16]);
        let h = self.b.op(op::U_CONVERT, u16t, &[h]);
        self.write(dst, h, u16t, false)
    }

    // ------------------------------------------------------------ atomics

    fn atom(&mut self, i: &Inst) -> R<()> {
        let t = inst_type(i)?;
        let ty = self.ty_of(t);
        let space = space_of(i);
        let is_red = i.op == "red";
        let (pa, va) = if is_red { (0, 1) } else { (1, 2) };
        let ptr = self.address(arg(i, pa)?, space, ty)?;
        let scope = self.b.const_u32(scope_of(i));
        let storage_bits = match space {
            StateSpace::Shared => 0x100,
            StateSpace::Global => 0x200,
            _ => 0x300,
        };
        let sem_bits = if i.has("acquire") {
            0x2
        } else if i.has("release") {
            0x4
        } else if i.has("acq_rel") {
            0x8
        } else {
            0
        };
        let sem = self.b.const_u32(if sem_bits == 0 { 0 } else { sem_bits | storage_bits });
        let opname = i
            .mods
            .iter()
            .find(|m| matches!(m.as_str(), "add" | "min" | "max" | "inc" | "dec" | "and" | "or" | "xor" | "exch" | "cas"))
            .ok_or("atom needs an operation")?
            .clone();
        if t.size() == 8 {
            self.b.capability(cap::INT64_ATOMICS);
        }
        let signed = t.is_signed();
        let v = self.read(arg(i, va)?, ty, signed)?;
        let r = match (opname.as_str(), t.is_float()) {
            ("cas", _) => {
                let c = self.read(arg(i, va + 1)?, ty, false)?;
                let (it, ptr_i) = self.int_view(ty, ptr, space)?;
                let vi = self.coerce(v, ty, it, false);
                let ci = self.coerce(c, ty, it, false);
                let r = self.b.op(op::ATOMIC_COMPARE_EXCHANGE, it, &[ptr_i, scope, sem, sem, ci, vi]);
                self.coerce(r, it, ty, false)
            }
            ("exch", _) => {
                let (it, ptr_i) = self.int_view(ty, ptr, space)?;
                let vi = self.coerce(v, ty, it, false);
                let r = self.b.op(op::ATOMIC_EXCHANGE, it, &[ptr_i, scope, sem, vi]);
                self.coerce(r, it, ty, false)
            }
            ("add", true) if self.opts.float_atomics && t != Scalar::F16 => {
                self.b.extension("SPV_EXT_shader_atomic_float_add");
                self.b.capability(if t == Scalar::F64 { cap::ATOMIC_FLOAT64_ADD_EXT } else { cap::ATOMIC_FLOAT32_ADD_EXT });
                self.b.op(op::ATOMIC_F_ADD_EXT, ty, &[ptr, scope, sem, v])
            }
            ("add" | "min" | "max", true) => {
                let o = opname.clone();
                self.cas_loop(ty, ptr, space, scope, sem, move |c, old| match o.as_str() {
                    "add" => c.b.op(op::F_ADD, ty, &[old, v]),
                    "min" => c.b.ext(ty, cl::FMIN, &[old, v]),
                    _ => c.b.ext(ty, cl::FMAX, &[old, v]),
                })?
            }
            ("inc" | "dec", false) => {
                let inc = opname == "inc";
                self.cas_loop(ty, ptr, space, scope, sem, move |c, old| {
                    let bt = c.b.bool();
                    let one = c.b.const_bits(ty, 1);
                    let zero = c.b.const_bits(ty, 0);
                    if inc {
                        // (old >= v) ? 0 : old + 1
                        let ge = c.b.op(op::U_GREATER_THAN_EQUAL, bt, &[old, v]);
                        let n = c.b.op(op::I_ADD, ty, &[old, one]);
                        c.b.op(op::SELECT, ty, &[ge, zero, n])
                    } else {
                        // (old == 0 || old > v) ? v : old - 1
                        let z = c.b.op(op::I_EQUAL, bt, &[old, zero]);
                        let gt = c.b.op(op::U_GREATER_THAN, bt, &[old, v]);
                        let w = c.b.op(op::LOGICAL_OR, bt, &[z, gt]);
                        let n = c.b.op(op::I_SUB, ty, &[old, one]);
                        c.b.op(op::SELECT, ty, &[w, v, n])
                    }
                })?
            }
            (o, false) => {
                let opc = match o {
                    "add" => op::ATOMIC_I_ADD,
                    "min" => if signed { op::ATOMIC_S_MIN } else { op::ATOMIC_U_MIN },
                    "max" => if signed { op::ATOMIC_S_MAX } else { op::ATOMIC_U_MAX },
                    "and" => op::ATOMIC_AND,
                    "or" => op::ATOMIC_OR,
                    "xor" => op::ATOMIC_XOR,
                    _ => return Err(format!("unsupported atomic {o}")),
                };
                self.b.op(opc, ty, &[ptr, scope, sem, v])
            }
            (o, true) => return Err(format!("unsupported float atomic {o}")),
        };
        if !is_red {
            self.write(arg(i, 0)?, r, ty, signed)?;
        }
        Ok(())
    }

    /// Reinterpret a pointer to a float as a pointer to a same-width integer.
    fn int_view(&mut self, ty: Id, ptr: Id, space: StateSpace) -> R<(Id, Id)> {
        if !self.is_float_ty(ty) {
            return Ok((ty, ptr));
        }
        let w = self.b.width(ty);
        let it = self.b.int(w);
        let pt = self.b.ptr(storage_of(space), it);
        Ok((it, self.b.op(op::BITCAST, pt, &[ptr])))
    }

    /// Compare-and-swap loop applying `f(old) -> new`; returns the old value.
    fn cas_loop(
        &mut self,
        ty: Id,
        ptr: Id,
        space: StateSpace,
        scope: Id,
        sem: Id,
        f: impl Fn(&mut Self, Id) -> Id,
    ) -> R<Id> {
        let (it, iptr) = self.int_view(ty, ptr, space)?;
        let bt = self.b.bool();
        let init = self.b.op(op::LOAD, it, &[iptr]);
        let pre = self.f.cur_block;
        let loop_l = self.b.id();
        let done = self.b.id();
        self.b.emit(op::BRANCH, &[loop_l]);
        self.b.emit(op::LABEL, &[loop_l]);
        self.f.cur_block = loop_l;
        let old = self.b.id();
        let cur = self.b.id();
        // phi placeholder emitted manually so we can reference `cur` before definition
        self.b.emit(op::PHI, &[it, old, init, pre, cur, loop_l]);
        let oldv = self.coerce(old, it, ty, false);
        let newv = f(self, oldv);
        let newi = self.coerce(newv, ty, it, false);
        self.b.emit(op::ATOMIC_COMPARE_EXCHANGE, &[it, cur, iptr, scope, sem, sem, newi, old]);
        let ok = self.b.op(op::I_EQUAL, bt, &[cur, old]);
        self.b.emit(op::BRANCH_CONDITIONAL, &[ok, done, loop_l]);
        self.b.emit(op::LABEL, &[done]);
        self.f.cur_block = done;
        Ok(self.coerce(old, it, ty, false))
    }

    fn bar_red(&mut self, i: &Inst) -> R<()> {
        Err(format!("bar.red is not supported yet ({:?})", i.mods))
    }

    // --------------------------------------------------------- warp ops

    fn ballot(&mut self, pred: Id) -> R<Id> {
        self.b.capability(cap::GROUP_NON_UNIFORM);
        self.b.capability(cap::GROUP_NON_UNIFORM_BALLOT);
        let u32t = self.b.int(32);
        let v4 = self.b.vector(u32t, 4);
        let sg = self.b.const_u32(3);
        let b = self.b.op(op::GROUP_NON_UNIFORM_BALLOT, v4, &[sg, pred]);
        Ok(self.b.op(op::COMPOSITE_EXTRACT, u32t, &[b, 0]))
    }

    fn vote(&mut self, i: &Inst) -> R<()> {
        self.f.uses_warp_ops = true;
        let bt = self.b.bool();
        let u32t = self.b.int(32);
        // vote{.sync}.mode.pred d, {!}a{, membermask}
        let a = self.read(arg(i, 1)?, bt, false)?;
        let sg = self.b.const_u32(3);
        if i.has("ballot") {
            let m = self.ballot(a)?;
            return self.write(arg(i, 0)?, m, u32t, false);
        }
        self.b.capability(cap::GROUP_NON_UNIFORM);
        self.b.capability(cap::GROUP_NON_UNIFORM_VOTE);
        let r = if i.has("all") {
            self.b.op(op::GROUP_NON_UNIFORM_ALL, bt, &[sg, a])
        } else if i.has("any") {
            self.b.op(op::GROUP_NON_UNIFORM_ANY, bt, &[sg, a])
        } else if i.has("uni") {
            let all = self.b.op(op::GROUP_NON_UNIFORM_ALL, bt, &[sg, a]);
            let na = self.b.op(op::LOGICAL_NOT, bt, &[a]);
            let none = self.b.op(op::GROUP_NON_UNIFORM_ALL, bt, &[sg, na]);
            self.b.op(op::LOGICAL_OR, bt, &[all, none])
        } else {
            return Err("unknown vote mode".into());
        };
        self.write(arg(i, 0)?, r, bt, false)
    }

    /// shfl.sync.{up,down,bfly,idx}.b32 d[|p], a, b, c, membermask
    fn shfl(&mut self, i: &Inst) -> R<()> {
        self.f.uses_warp_ops = true;
        let u32t = self.b.int(32);
        let bt = self.b.bool();
        let sg = self.b.const_u32(3);
        let a = self.read(arg(i, 1)?, u32t, false)?;
        let b = self.read(arg(i, 2)?, u32t, false)?;
        let c = self.read(arg(i, 3)?, u32t, false)?;
        let lane_var = self.builtin(builtin::SUBGROUP_LOCAL_INVOCATION_ID, u32t);
        let lane = self.b.op(op::LOAD, u32t, &[lane_var]);
        // c packs clamp (bits 0-4) and segment mask (bits 8-12)
        let x1f = self.b.const_u32(0x1f);
        let b = self.b.op(op::BITWISE_AND, u32t, &[b, x1f]);
        let clamp = self.b.op(op::BITWISE_AND, u32t, &[c, x1f]);
        let s8 = self.b.const_u32(8);
        let segmask = self.b.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[c, s8]);
        let segmask = self.b.op(op::BITWISE_AND, u32t, &[segmask, x1f]);
        let notseg = self.b.op(op::NOT, u32t, &[segmask]);
        let min_l = self.b.op(op::BITWISE_AND, u32t, &[lane, segmask]);
        let cl_ns = self.b.op(op::BITWISE_AND, u32t, &[clamp, notseg]);
        let max_l = self.b.op(op::BITWISE_OR, u32t, &[min_l, cl_ns]);
        let (j, valid) = if i.has("up") {
            let j = self.b.op(op::I_SUB, u32t, &[lane, b]);
            let v = self.b.op(op::S_GREATER_THAN_EQUAL, bt, &[j, max_l]);
            (j, v)
        } else if i.has("down") {
            let j = self.b.op(op::I_ADD, u32t, &[lane, b]);
            let v = self.b.op(op::S_LESS_THAN_EQUAL, bt, &[j, max_l]);
            (j, v)
        } else if i.has("bfly") {
            let j = self.b.op(op::BITWISE_XOR, u32t, &[lane, b]);
            let v = self.b.op(op::S_LESS_THAN_EQUAL, bt, &[j, max_l]);
            (j, v)
        } else {
            let nseg = self.b.op(op::BITWISE_AND, u32t, &[b, notseg]);
            let j = self.b.op(op::BITWISE_OR, u32t, &[min_l, nseg]);
            let v = self.b.op(op::S_LESS_THAN_EQUAL, bt, &[j, max_l]);
            (j, v)
        };
        let src = self.b.op(op::SELECT, u32t, &[valid, j, lane]);
        self.b.capability(cap::GROUP_NON_UNIFORM);
        self.b.capability(cap::GROUP_NON_UNIFORM_SHUFFLE);
        let r = self.b.op(op::GROUP_NON_UNIFORM_SHUFFLE, u32t, &[sg, a, src]);
        match arg(i, 0)? {
            Operand::Pair(d, p) => {
                self.write(d, r, u32t, false)?;
                self.write(p, valid, bt, false)
            }
            d => self.write(d, r, u32t, false),
        }
    }

    // -------------------------------------------------------------- calls

    fn call(&mut self, i: &Inst) -> R<()> {
        // call{.uni} (ret...), func, (args...)  |  call func, (args)  |  call func
        let mut it = i.args.iter().peekable();
        let rets: Vec<Operand> = match it.peek() {
            Some(Operand::List(l)) => {
                let l = l.clone();
                it.next();
                l
            }
            _ => vec![],
        };
        let callee = it.next().and_then(|o| o.id()).ok_or("call needs a target")?.to_string();
        let args: Vec<Operand> = match it.next() {
            Some(Operand::List(l)) => l.clone(),
            None => vec![],
            Some(o) => return Err(format!("unexpected call operand {o:?}")),
        };
        if !self.funcs.contains_key(&callee) {
            return Err(format!("indirect call through {callee} is not supported yet"));
        }
        if is_runtime_builtin(&callee) && !self.funcs_has_body(&callee) {
            return self.runtime_call(&callee, &rets, &args);
        }
        let (fid, ret_ty, ptys, rtys) = {
            let s = &self.funcs[&callee];
            (s.id, s.ret_ty, s.params.iter().map(|p| p.1).collect::<Vec<_>>(), s.rets.iter().map(|r| r.1).collect::<Vec<_>>())
        };
        let mut vals = Vec::new();
        for (a, &t) in args.iter().zip(&ptys) {
            vals.push(self.read_param(a, t)?);
        }
        if let Some(d) = self.f.dyn_smem {
            vals.push(d);
        }
        let mut ops = vec![fid];
        ops.extend(vals);
        let r = self.b.op(op::FUNCTION_CALL, ret_ty, &ops);
        match rtys.len() {
            0 => {}
            1 => self.write_param(&rets[0], r, rtys[0])?,
            _ => {
                for (k, (d, &t)) in rets.iter().zip(&rtys).enumerate() {
                    let x = self.b.op(op::COMPOSITE_EXTRACT, t, &[r, k as u32]);
                    self.write_param(d, x, t)?;
                }
            }
        }
        Ok(())
    }

    fn funcs_has_body(&self, _name: &str) -> bool {
        false
    }

    /// Value of a call argument: a `.param` variable (loaded whole) or a register/immediate.
    fn read_param(&mut self, a: &Operand, t: Id) -> R<Id> {
        if let Some(n) = a.id() {
            if let Some(Var::Mem(s)) = self.lookup(n) {
                let p = self.sym_ptr(&s)?;
                let pt = self.b.ptr(s.storage, t);
                let p = self.b.op(op::BITCAST, pt, &[p]);
                return Ok(self.b.op(op::LOAD, t, &[p]));
            }
        }
        self.read(a, t, false)
    }

    fn write_param(&mut self, d: &Operand, v: Id, t: Id) -> R<()> {
        if let Some(n) = d.id() {
            if let Some(Var::Mem(s)) = self.lookup(n) {
                let p = self.sym_ptr(&s)?;
                let pt = self.b.ptr(s.storage, t);
                let p = self.b.op(op::BITCAST, pt, &[p]);
                self.b.emit(op::STORE, &[p, v]);
                return Ok(());
            }
        }
        self.write(d, v, t, false)
    }

    /// Calls into the CUDA device runtime (`vprintf`, `__assertfail`, `malloc`, `free`).
    fn runtime_call(&mut self, name: &str, rets: &[Operand], args: &[Operand]) -> R<()> {
        let _ = args;
        match name {
            "vprintf" => {
                // TODO: forward to OpenCL printf (needs format string in UniformConstant)
                let u32t = self.b.int(32);
                let z = self.b.const_u32(0);
                if let Some(r) = rets.first() {
                    self.write_param(r, z, u32t)?;
                }
                Ok(())
            }
            "__assertfail" | "__assert_fail" => Ok(()),
            _ => Err(format!("device runtime function {name} is not supported yet")),
        }
    }
}

fn is_runtime_builtin(name: &str) -> bool {
    matches!(name, "vprintf" | "__assertfail" | "__assert_fail" | "malloc" | "free")
}

fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        let r = (m + 0x1000 + ((m >> 13) & 1) - 1) >> 13;
        return sign | r as u16;
    }
    let r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) { r + 1 } else { r };
    sign | r as u16
}
