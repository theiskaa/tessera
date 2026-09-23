//! Minimal JSON reader for the safetensors header. It supports exactly what the header needs:
//! objects, arrays, strings with standard escapes, non-negative integers, and literals. It is
//! not a general parser, and every out-of-range read becomes `BundleInvalid` rather than a panic.
//!
//! Numbers are integers only because every number in a header is an offset, a dimension, or a
//! feature setting; parsing `f64` would pull float parsing into the wasm build, about 15 KB gzip.

use crate::Error;

/// A parsed JSON value. Objects keep their keys in file order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(u64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Header documents are shallow; this bound keeps adversarial nesting from exhausting the stack.
const MAX_DEPTH: usize = 32;
/// Keys per object. A header has a few dozen; the bound keeps the duplicate check cheap.
const MAX_KEYS: usize = 1024;

impl Json {
    pub(crate) fn parse(text: &str) -> Result<Json, Error> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
            depth: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(Error::BundleInvalid);
        }
        Ok(v)
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn entries(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(kv) => kv,
            _ => &[],
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub(crate) fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|n| usize::try_from(n).ok())
    }

    pub(crate) fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn next(&mut self) -> Result<u8, Error> {
        let b = self.peek().ok_or(Error::BundleInvalid)?;
        self.i += 1;
        Ok(b)
    }

    fn expect(&mut self, b: u8) -> Result<(), Error> {
        if self.next()? == b {
            Ok(())
        } else {
            Err(Error::BundleInvalid)
        }
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, Error> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'0'..=b'9') => self.number(),
            _ => Err(Error::BundleInvalid),
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Json, Error>) -> Result<Json, Error> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Error::BundleInvalid);
        }
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn object(&mut self) -> Result<Json, Error> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.ws();
            let key = self.string()?;
            if out.len() == MAX_KEYS || out.iter().any(|(k, _)| *k == key) {
                return Err(Error::BundleInvalid);
            }
            self.ws();
            self.expect(b':')?;
            let v = self.value()?;
            out.push((key, v));
            self.ws();
            match self.next()? {
                b',' => continue,
                b'}' => return Ok(Json::Obj(out)),
                _ => return Err(Error::BundleInvalid),
            }
        }
    }

    fn array(&mut self) -> Result<Json, Error> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.next()? {
                b',' => continue,
                b']' => return Ok(Json::Arr(out)),
                _ => return Err(Error::BundleInvalid),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = (self.next()? as char)
                .to_digit(16)
                .ok_or(Error::BundleInvalid)?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            match self.next()? {
                b'"' => return String::from_utf8(out).map_err(|_| Error::BundleInvalid),
                b'\\' => {
                    let c = match self.next()? {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(Error::BundleInvalid);
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(code).ok_or(Error::BundleInvalid)?
                        }
                        _ => return Err(Error::BundleInvalid),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                b if b < 0x20 => return Err(Error::BundleInvalid),
                b => out.push(b),
            }
        }
    }

    fn number(&mut self) -> Result<Json, Error> {
        if self.peek() == Some(b'0') && matches!(self.s.get(self.i + 1), Some(b'0'..=b'9')) {
            return Err(Error::BundleInvalid);
        }
        let mut n: u64 = 0;
        while let Some(d @ b'0'..=b'9') = self.peek() {
            n = n
                .checked_mul(10)
                .and_then(|n| n.checked_add(u64::from(d - b'0')))
                .ok_or(Error::BundleInvalid)?;
            self.i += 1;
        }
        // A fraction or exponent means a value no header field holds.
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(Error::BundleInvalid);
        }
        Ok(Json::Num(n))
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, Error> {
        if self.s.get(self.i..self.i + word.len()) == Some(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(Error::BundleInvalid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values() {
        let j =
            Json::parse(r#"{"a": [1, 25, 300], "b": {"c": null, "d": true}, "e": "x"}"#).unwrap();
        assert_eq!(
            j.get("a").and_then(Json::as_arr).map(<[Json]>::len),
            Some(3)
        );
        assert_eq!(j.get("e").and_then(Json::as_str), Some("x"));
        assert_eq!(j.get("b").and_then(|b| b.get("d")), Some(&Json::Bool(true)));
    }

    #[test]
    fn decodes_escapes_and_surrogate_pairs() {
        assert_eq!(Json::parse(r#""\u00e9""#).unwrap(), Json::Str("é".into()));
        assert_eq!(
            Json::parse(r#""\ud83d\ude00""#).unwrap(),
            Json::Str("😀".into())
        );
        assert_eq!(
            Json::parse(r#""a\"b\\c\n""#).unwrap(),
            Json::Str("a\"b\\c\n".into())
        );
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "[1,",
            "{\"a\" 1}",
            "\"abc",
            "tru",
            "{\"a\":1,}",
            "[1] 2",
            "\"\\ud83d\"",
            "{\"a\":1,\"a\":2}",
        ] {
            assert_eq!(Json::parse(bad), Err(Error::BundleInvalid), "{bad}");
        }
    }

    #[test]
    fn deep_nesting_is_rejected_not_overflowed() {
        let deep = "[".repeat(10_000);
        assert_eq!(Json::parse(&deep), Err(Error::BundleInvalid));
    }

    #[test]
    fn integers_as_usize() {
        assert_eq!(Json::parse("42").unwrap().as_usize(), Some(42));
        for bad in ["-1", "1.5", "1e3", "99999999999999999999", "007"] {
            assert_eq!(Json::parse(bad), Err(Error::BundleInvalid), "{bad}");
        }
    }
}
