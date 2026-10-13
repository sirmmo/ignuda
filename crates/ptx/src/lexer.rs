#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    /// A dot-word: `.global` -> Dir("global")
    Dir(String),
    /// Raw numeric literal text (integer, decimal float, or 0f/0d hex float).
    Num(String),
    Str(String),
    Punct(char),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
}

fn is_id_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$' || c == b'%'
}
fn is_id_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'$'
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut line = 1u32;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                if b[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i += 2;
        } else if c == b'"' {
            let start = i + 1;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            out.push(Token { tok: Tok::Str(src[start..i.min(b.len())].to_string()), line });
            i += 1;
        } else if c == b'.' && i + 1 < b.len() && is_id_char(b[i + 1]) && {
            // `.2d`, `.3d` are dot-words; `.5` is not
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            j == i + 1 || (j < b.len() && b[j].is_ascii_alphabetic())
        } {
            let start = i + 1;
            i += 1;
            while i < b.len() && is_id_char(b[i]) {
                i += 1;
            }
            out.push(Token { tok: Tok::Dir(src[start..i].to_string()), line });
        } else if c.is_ascii_digit() {
            let start = i;
            if c == b'0' && matches!(b.get(i + 1), Some(b'f' | b'F' | b'd' | b'D' | b'x' | b'X')) {
                i += 2;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
            } else if c == b'0' && matches!(b.get(i + 1), Some(b'b' | b'B')) {
                i += 2;
                while i < b.len() && (b[i] == b'0' || b[i] == b'1') {
                    i += 1;
                }
            } else {
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'.' && b.get(i + 1).map_or(false, |c| c.is_ascii_digit()) {
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    let save = i;
                    i += 1;
                    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                        i += 1;
                    }
                    if i < b.len() && b[i].is_ascii_digit() {
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    } else {
                        i = save;
                    }
                }
            }
            if i < b.len() && (b[i] == b'U' || b[i] == b'u') {
                i += 1;
            }
            out.push(Token { tok: Tok::Num(src[start..i].to_string()), line });
        } else if is_id_start(c) {
            let start = i;
            i += 1;
            while i < b.len() && is_id_char(b[i]) {
                i += 1;
            }
            out.push(Token { tok: Tok::Ident(src[start..i].to_string()), line });
        } else if b",;:{}[]()<>+-!|@=*".contains(&c) {
            out.push(Token { tok: Tok::Punct(c as char), line });
            i += 1;
        } else {
            return Err(format!("line {line}: unexpected character {:?}", c as char));
        }
    }
    Ok(out)
}

/// Parse an integer literal as PTX writes it (decimal, 0x hex, 0b binary,
/// leading-0 octal, optional U suffix).
pub fn parse_int(s: &str) -> Option<i128> {
    let s = s.trim_end_matches(['U', 'u']);
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i128::from_str_radix(h, 16).ok()
    } else if let Some(h) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i128::from_str_radix(h, 2).ok()
    } else if s.len() > 1 && s.starts_with('0') && s.bytes().all(|c| c.is_ascii_digit()) {
        i128::from_str_radix(&s[1..], 8).ok()
    } else {
        s.parse().ok()
    }
}
