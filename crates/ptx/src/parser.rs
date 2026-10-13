use crate::ast::*;
use crate::lexer::{lex, parse_int, Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

pub type PResult<T> = Result<T, String>;

pub fn parse(src: &str) -> PResult<Module> {
    let toks = lex(src)?;
    Parser { toks, pos: 0 }.module()
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }
    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n).map(|t| &t.tok)
    }
    fn line(&self) -> u32 {
        self.toks.get(self.pos).or(self.toks.last()).map_or(0, |t| t.line)
    }
    fn next(&mut self) -> PResult<Tok> {
        let t = self.toks.get(self.pos).map(|t| t.tok.clone()).ok_or("unexpected end of input")?;
        self.pos += 1;
        Ok(t)
    }
    fn err<T>(&self, msg: impl std::fmt::Display) -> PResult<T> {
        Err(format!("line {}: {msg} (at {:?})", self.line(), self.peek()))
    }
    fn is_punct(&self, c: char) -> bool {
        self.peek() == Some(&Tok::Punct(c))
    }
    fn eat_punct(&mut self, c: char) -> bool {
        if self.is_punct(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect_punct(&mut self, c: char) -> PResult<()> {
        if self.eat_punct(c) {
            Ok(())
        } else {
            self.err(format!("expected '{c}'"))
        }
    }
    fn is_dir(&self, d: &str) -> bool {
        matches!(self.peek(), Some(Tok::Dir(x)) if x == d)
    }
    fn eat_dir(&mut self, d: &str) -> bool {
        if self.is_dir(d) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn ident(&mut self) -> PResult<String> {
        match self.next()? {
            Tok::Ident(s) => Ok(s),
            t => Err(format!("line {}: expected identifier, got {t:?}", self.line())),
        }
    }
    fn int(&mut self) -> PResult<i128> {
        let neg = self.eat_punct('-');
        match self.next()? {
            Tok::Num(s) => {
                parse_int(&s).map(|v| if neg { -v } else { v }).ok_or(format!("bad integer {s}"))
            }
            t => Err(format!("line {}: expected integer, got {t:?}", self.line())),
        }
    }
    fn skip_to_semicolon(&mut self) -> PResult<()> {
        while !self.eat_punct(';') {
            self.next()?;
        }
        Ok(())
    }

    fn module(mut self) -> PResult<Module> {
        let mut m = Module { address_size: 64, ..Default::default() };
        while let Some(t) = self.peek().cloned() {
            match t {
                Tok::Dir(d) => match d.as_str() {
                    "version" => {
                        self.pos += 1;
                        if let Tok::Num(v) = self.next()? {
                            let mut it = v.split('.');
                            m.version.0 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                            m.version.1 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                        }
                    }
                    "target" => {
                        self.pos += 1;
                        m.target.push(self.ident()?);
                        while self.eat_punct(',') {
                            m.target.push(self.ident()?);
                        }
                    }
                    "address_size" => {
                        self.pos += 1;
                        m.address_size = self.int()? as u32;
                    }
                    "file" => {
                        self.pos += 1;
                        self.int()?;
                        self.next()?; // filename
                        while self.eat_punct(',') {
                            self.int()?;
                        }
                    }
                    "section" => {
                        self.pos += 1;
                        self.next()?;
                        self.skip_braces()?;
                    }
                    "pragma" | "alias" => {
                        self.pos += 1;
                        self.skip_to_semicolon()?;
                    }
                    _ => {
                        let linkage = self.linkage();
                        if self.eat_dir("entry") {
                            m.items.push(Item::Func(self.function(FuncKind::Entry, linkage)?));
                        } else if self.eat_dir("func") {
                            m.items.push(Item::Func(self.function(FuncKind::Func, linkage)?));
                        } else {
                            for mut v in self.var_decl()? {
                                v.linkage = linkage;
                                m.items.push(Item::Var(v));
                            }
                        }
                    }
                },
                _ => return self.err("unexpected token at module level"),
            }
        }
        Ok(m)
    }

    fn skip_braces(&mut self) -> PResult<()> {
        self.expect_punct('{')?;
        let mut depth = 1;
        while depth > 0 {
            match self.next()? {
                Tok::Punct('{') => depth += 1,
                Tok::Punct('}') => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    fn linkage(&mut self) -> Linkage {
        let mut l = Linkage::None;
        loop {
            l = if self.eat_dir("visible") {
                Linkage::Visible
            } else if self.eat_dir("extern") {
                Linkage::Extern
            } else if self.eat_dir("weak") {
                Linkage::Weak
            } else if self.eat_dir("common") {
                Linkage::Common
            } else {
                return l;
            }
        }
    }

    /// Parses `.space [.align N] [.ptr ...] [.vN] .type name[<N>|[dims]] [= init] {, name...} ;`
    /// When `in_params` is set, a single declaration is parsed without the trailing `;`.
    fn var_decl_inner(&mut self, in_params: bool) -> PResult<Vec<VarDecl>> {
        let line = self.line();
        let space = match self.next()? {
            Tok::Dir(d) => StateSpace::from_dir(&d).ok_or(format!("line {line}: unknown state space .{d}"))?,
            t => return Err(format!("line {line}: expected state space, got {t:?}")),
        };
        let mut align = None;
        let mut vector = 0u8;
        let mut scalar = None;
        loop {
            match self.peek() {
                Some(Tok::Dir(d)) => {
                    let d = d.clone();
                    self.pos += 1;
                    match d.as_str() {
                        "align" => align = Some(self.int()? as u32),
                        "v2" => vector = 2,
                        "v4" => vector = 4,
                        "v8" => vector = 8,
                        "ptr" | "global" | "const" | "local" | "shared" => {} // .ptr attributes
                        _ => {
                            scalar = Some(Scalar::from_dir(&d).ok_or(format!("line {line}: unknown type .{d}"))?)
                        }
                    }
                }
                _ => break,
            }
        }
        let ty = Type { scalar: scalar.ok_or(format!("line {line}: missing type"))?, vector };
        let mut out = Vec::new();
        loop {
            let name = self.ident()?;
            let mut v = VarDecl { space, linkage: Linkage::None, align, ty, name, dims: vec![], count: None, init: None, line };
            if self.eat_punct('<') {
                v.count = Some(self.int()? as u32);
                self.expect_punct('>')?;
            }
            while self.eat_punct('[') {
                if self.eat_punct(']') {
                    v.dims.push(None);
                } else {
                    v.dims.push(Some(self.int()? as u64));
                    self.expect_punct(']')?;
                }
            }
            if self.eat_punct('=') {
                v.init = Some(self.init()?);
            }
            out.push(v);
            if in_params || !self.eat_punct(',') {
                break;
            }
        }
        if !in_params {
            self.expect_punct(';')?;
        }
        Ok(out)
    }

    fn var_decl(&mut self) -> PResult<Vec<VarDecl>> {
        self.var_decl_inner(false)
    }

    fn init(&mut self) -> PResult<Init> {
        if self.eat_punct('{') {
            let mut items = Vec::new();
            if !self.is_punct('}') {
                loop {
                    items.push(self.init()?);
                    if !self.eat_punct(',') {
                        break;
                    }
                }
            }
            self.expect_punct('}')?;
            Ok(Init::List(items))
        } else {
            Ok(Init::Expr(self.expr()?))
        }
    }

    fn expr(&mut self) -> PResult<Expr> {
        match self.peek().cloned() {
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                let (name, generic) = if name == "generic" && self.eat_punct('(') {
                    let n = self.ident()?;
                    self.expect_punct(')')?;
                    (n, true)
                } else {
                    (name, false)
                };
                let mut offset = 0;
                if self.eat_punct('+') {
                    offset = self.int()? as i64;
                }
                Ok(Expr::Sym { name, offset, generic })
            }
            _ => match self.number_operand()? {
                Operand::Int(v) => Ok(Expr::Int(v)),
                Operand::F32(v) => Ok(Expr::F32(v)),
                Operand::F64(v) => Ok(Expr::F64(v)),
                _ => unreachable!(),
            },
        }
    }

    fn number_operand(&mut self) -> PResult<Operand> {
        let neg = self.eat_punct('-');
        let s = match self.next()? {
            Tok::Num(s) => s,
            t => return Err(format!("line {}: expected number, got {t:?}", self.line())),
        };
        let lower = s.to_ascii_lowercase();
        if let Some(h) = lower.strip_prefix("0f") {
            let v = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
            return Ok(Operand::F32(if neg { v ^ 0x8000_0000 } else { v }));
        }
        if let Some(h) = lower.strip_prefix("0d") {
            let v = u64::from_str_radix(h, 16).map_err(|e| e.to_string())?;
            return Ok(Operand::F64(if neg { v ^ (1 << 63) } else { v }));
        }
        if !lower.starts_with("0x") && (lower.contains('.') || lower.contains('e')) {
            let v: f64 = lower.parse().map_err(|_| format!("bad float {s}"))?;
            return Ok(Operand::F64((if neg { -v } else { v }).to_bits()));
        }
        let v = parse_int(&s).ok_or(format!("bad integer {s}"))?;
        Ok(Operand::Int(if neg { -v } else { v }))
    }

    fn param_list(&mut self) -> PResult<Vec<VarDecl>> {
        let mut ps = Vec::new();
        self.expect_punct('(')?;
        if !self.eat_punct(')') {
            loop {
                ps.extend(self.var_decl_inner(true)?);
                if !self.eat_punct(',') {
                    break;
                }
            }
            self.expect_punct(')')?;
        }
        Ok(ps)
    }

    fn dims3(&mut self) -> PResult<[u32; 3]> {
        let mut d = [1u32; 3];
        d[0] = self.int()? as u32;
        for x in d.iter_mut().skip(1) {
            if !self.eat_punct(',') {
                break;
            }
            *x = self.int()? as u32;
        }
        Ok(d)
    }

    fn function(&mut self, kind: FuncKind, linkage: Linkage) -> PResult<Function> {
        let line = self.line();
        let ret = if self.is_punct('(') { self.param_list()? } else { vec![] };
        let name = self.ident()?;
        let params = if self.is_punct('(') { self.param_list()? } else { vec![] };
        let mut f = Function {
            kind,
            linkage,
            name,
            ret,
            params,
            body: None,
            maxntid: None,
            reqntid: None,
            noreturn: false,
            line,
        };
        loop {
            match self.peek().cloned() {
                Some(Tok::Dir(d)) => {
                    self.pos += 1;
                    match d.as_str() {
                        "maxntid" => f.maxntid = Some(self.dims3()?),
                        "reqntid" => f.reqntid = Some(self.dims3()?),
                        "noreturn" => f.noreturn = true,
                        "minnctapersm" | "maxnreg" | "maxclusterrank" => {
                            self.int()?;
                        }
                        "reqnctapercluster" => {
                            self.dims3()?;
                        }
                        "explicitcluster" => {}
                        "pragma" => self.skip_to_semicolon()?,
                        _ => return self.err(format!("unknown function directive .{d}")),
                    }
                }
                Some(Tok::Punct(';')) => {
                    self.pos += 1;
                    return Ok(f);
                }
                Some(Tok::Punct('{')) => {
                    self.pos += 1;
                    f.body = Some(self.block()?);
                    return Ok(f);
                }
                _ => return self.err("expected function body"),
            }
        }
    }

    /// Statements until the matching `}` (which is consumed).
    fn block(&mut self) -> PResult<Vec<Stmt>> {
        let mut out = Vec::new();
        loop {
            let line = self.line();
            match self.peek().cloned() {
                None => return self.err("unterminated block"),
                Some(Tok::Punct('}')) => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(Tok::Punct('{')) => {
                    self.pos += 1;
                    out.push(Stmt::Block(self.block()?));
                }
                Some(Tok::Dir(d)) => match d.as_str() {
                    "loc" | "pragma" => {
                        self.pos += 1;
                        if d == "pragma" {
                            self.skip_to_semicolon()?;
                        } else {
                            // .loc file line col [, inlined_at ...] — no terminator
                            // `function_name $L__info_string0, inlined_at 1 2 3` attributes
                            // stay on the same line.
                            while self.toks.get(self.pos).map_or(false, |t| {
                                t.line == line && matches!(t.tok, Tok::Num(_) | Tok::Ident(_) | Tok::Punct(',' | '+'))
                            }) {
                                self.pos += 1;
                            }
                        }
                    }
                    "file" => {
                        self.pos += 1;
                        self.int()?;
                        self.next()?;
                    }
                    _ => {
                        let linkage = self.linkage();
                        for mut v in self.var_decl()? {
                            v.linkage = linkage;
                            out.push(Stmt::Decl(v));
                        }
                    }
                },
                Some(Tok::Ident(name)) if self.peek_at(1) == Some(&Tok::Punct(':')) => {
                    self.pos += 2;
                    if self.eat_dir("callprototype") {
                        let ret = if self.is_punct('(') { self.param_list()? } else { vec![] };
                        self.ident()?; // `_`
                        let params = if self.is_punct('(') { self.param_list()? } else { vec![] };
                        while !self.eat_punct(';') {
                            self.next()?; // .noreturn etc.
                        }
                        out.push(Stmt::CallProto { name, ret, params });
                    } else {
                        out.push(Stmt::Label(name));
                    }
                }
                _ => out.push(Stmt::Inst(self.inst()?)),
            }
        }
    }

    fn inst(&mut self) -> PResult<Inst> {
        let line = self.line();
        let mut pred = None;
        if self.eat_punct('@') {
            let neg = self.eat_punct('!');
            pred = Some((neg, self.ident()?));
        }
        let op = self.ident()?;
        let mut mods = Vec::new();
        while let Some(Tok::Dir(d)) = self.peek() {
            mods.push(d.clone());
            self.pos += 1;
        }
        let mut args = Vec::new();
        if !self.eat_punct(';') {
            loop {
                let mut a = self.operand()?;
                if self.eat_punct('|') {
                    a = Operand::Pair(Box::new(a), Box::new(self.operand()?));
                }
                args.push(a);
                if !self.eat_punct(',') {
                    break;
                }
            }
            if !self.eat_punct(';') {
                return self.err(format!("expected ';' after {op} operands"));
            }
        }
        Ok(Inst { pred, op, mods, args, line })
    }

    fn operand_list(&mut self, close: char) -> PResult<Vec<Operand>> {
        let mut v = Vec::new();
        if !self.eat_punct(close) {
            loop {
                v.push(self.operand()?);
                if !self.eat_punct(',') {
                    break;
                }
            }
            self.expect_punct(close)?;
        }
        Ok(v)
    }

    fn operand(&mut self) -> PResult<Operand> {
        match self.peek().cloned() {
            Some(Tok::Punct('!')) => {
                self.pos += 1;
                Ok(Operand::Not(Box::new(self.operand()?)))
            }
            Some(Tok::Punct('{')) => {
                self.pos += 1;
                Ok(Operand::Vector(self.operand_list('}')?))
            }
            Some(Tok::Punct('(')) => {
                self.pos += 1;
                Ok(Operand::List(self.operand_list(')')?))
            }
            Some(Tok::Punct('[')) => {
                self.pos += 1;
                self.address()
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if let Some(Tok::Dir(c)) = self.peek() {
                    let comp = match c.as_str() {
                        "x" | "r" => Some(0),
                        "y" | "g" => Some(1),
                        "z" | "b" => Some(2),
                        "w" | "a" => Some(3),
                        _ => None,
                    };
                    if comp.is_some() {
                        self.pos += 1;
                        return Ok(Operand::Id { name, comp });
                    }
                }
                if self.is_punct('+') && matches!(self.peek_at(1), Some(Tok::Num(_)) | Some(Tok::Punct('-'))) {
                    self.pos += 1;
                    let offset = self.int()? as i64;
                    return Ok(Operand::SymOff { name, offset });
                }
                Ok(Operand::Id { name, comp: None })
            }
            Some(Tok::Num(_)) | Some(Tok::Punct('-')) => self.number_operand(),
            _ => self.err("expected operand"),
        }
    }

    /// After `[`: `base`, `base+off`, `base+-off`, `off`, or `a, b, ...`.
    fn address(&mut self) -> PResult<Operand> {
        let mut base = None;
        let mut offset = 0i64;
        match self.peek().cloned() {
            Some(Tok::Ident(n)) => {
                self.pos += 1;
                if self.is_punct(',') {
                    // texture form: [tex, coords]
                    let mut items = vec![Operand::Id { name: n, comp: None }];
                    while self.eat_punct(',') {
                        items.push(self.operand()?);
                    }
                    self.expect_punct(']')?;
                    return Ok(Operand::AddrList(items));
                }
                base = Some(n);
                if self.eat_punct('+') {
                    offset = self.int()? as i64;
                } else if self.is_punct('-') {
                    offset = self.int()? as i64;
                }
            }
            _ => offset = self.int()? as i64,
        }
        self.expect_punct(']')?;
        Ok(Operand::Addr { base, offset })
    }
}
