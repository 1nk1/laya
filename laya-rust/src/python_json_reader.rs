//! A JSON reader that reads text the way Python's `json.loads` does, for the checkpoint files
//! Python parses with `json.load` (`rl_agent_config.json`, `tokenizer_config.json`).
//!
//! `serde_json` differs from Python in ways the checkpoint loaders can observe:
//! - it refuses the bare constants `NaN`, `Infinity` and `-Infinity`, which Python reads as floats;
//! - it refuses a float literal that overflows (`1e400`, which Python reads as `inf`);
//! - it turns an integer above `u64::MAX` (or `-0`) into a float, where Python keeps an `int`.
//!
//! [`PyJson`] keeps all of those apart from strings and from each other, so nothing is encoded
//! in-band: an integer keeps its literal text, a float is a correctly rounded `f64` (possibly
//! non-finite), and a string is always a string. Anything Python rejects is rejected here: the
//! constants are values only (never object keys), numbers follow the JSON grammar exactly, and
//! trailing commas or trailing data are errors. One known difference remains: a lone surrogate
//! escape (`"\ud800"`) loads in Python and is an error here, since a Rust `String` cannot hold one.

use crate::error::{LayaError, Result};
use serde::de::Error as _;
use serde_json::Value;

/// Python's recursion limit makes `json.loads` fail on arrays nested about a thousand deep.
const MAX_DEPTH: usize = 1000;

/// A parsed JSON value with Python's number kinds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PyJson {
    Null,
    Bool(bool),
    /// A JSON integer literal (no fraction or exponent), exactly as written: Python's `int`.
    Int(String),
    /// A JSON float, including `NaN`, `Infinity`, `-Infinity` and literals that overflow to `inf`.
    Float(f64),
    Str(String),
    Array(Vec<PyJson>),
    /// Entries in file order; a repeated key keeps its first position and its last value, as a
    /// Python `dict` does.
    Object(Vec<(String, PyJson)>),
}

impl PyJson {
    /// Parse `text` as `json.loads` does.
    pub(crate) fn parse(text: &str) -> Result<PyJson> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let outcome = reader
            .skip_ws()
            .and_then(|()| reader.value(0))
            .and_then(|value| {
                reader.skip_ws()?;
                if reader.pos == reader.bytes.len() {
                    Ok(value)
                } else {
                    Err("trailing characters after the JSON value".to_string())
                }
            });
        match outcome {
            Ok(value) => Ok(value),
            // Prefer serde_json's own message (line and column) when it also rejects the text.
            Err(message) => match serde_json::from_str::<Value>(text) {
                Err(e) => Err(LayaError::Json(e)),
                Ok(_) => Err(LayaError::Json(serde_json::Error::custom(format!(
                    "{message} at byte {}",
                    reader.pos
                )))),
            },
        }
    }

    /// The value of `key` when `self` is an object.
    pub(crate) fn get(&self, key: &str) -> Option<&PyJson> {
        match self {
            PyJson::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            PyJson::Str(s) => Some(s),
            _ => None,
        }
    }

    /// An integer literal as a `u64`, as `Value::as_u64` reads one (floats do not count).
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            PyJson::Int(literal) => literal.parse().ok(),
            _ => None,
        }
    }

    /// Python's `float(int)` for an integer literal: correctly rounded, `None` on overflow.
    pub(crate) fn int_as_f64(literal: &str) -> Option<f64> {
        literal.parse::<f64>().ok().filter(|f| f.is_finite())
    }

    /// A `serde_json::Value` view for code that only reads strings and objects. A non-finite
    /// number has no `Value` form and becomes `null`; either way it is not a string.
    pub(crate) fn to_value(&self) -> Value {
        match self {
            PyJson::Null => Value::Null,
            PyJson::Bool(b) => Value::Bool(*b),
            PyJson::Int(literal) => literal
                .parse::<i64>()
                .map(Value::from)
                .or_else(|_| literal.parse::<u64>().map(Value::from))
                .ok()
                .or_else(|| {
                    Self::int_as_f64(literal)
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                })
                .unwrap_or(Value::Null),
            PyJson::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            PyJson::Str(s) => Value::String(s.clone()),
            PyJson::Array(items) => Value::Array(items.iter().map(PyJson::to_value).collect()),
            PyJson::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_value()))
                    .collect(),
            ),
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn skip_ws(&mut self) -> std::result::Result<(), String> {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
        Ok(())
    }

    fn starts_with(&self, word: &str) -> bool {
        self.bytes[self.pos..].starts_with(word.as_bytes())
    }

    fn value(&mut self, depth: usize) -> std::result::Result<PyJson, String> {
        if depth > MAX_DEPTH {
            return Err("nested too deeply".to_string());
        }
        match self.bytes.get(self.pos) {
            Some(b'"') => self.string().map(PyJson::Str),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'n') if self.starts_with("null") => self.word("null", PyJson::Null),
            Some(b't') if self.starts_with("true") => self.word("true", PyJson::Bool(true)),
            Some(b'f') if self.starts_with("false") => self.word("false", PyJson::Bool(false)),
            Some(b'N') if self.starts_with("NaN") => self.word("NaN", PyJson::Float(f64::NAN)),
            Some(b'I') if self.starts_with("Infinity") => {
                self.word("Infinity", PyJson::Float(f64::INFINITY))
            }
            Some(b'-') if self.starts_with("-Infinity") => {
                self.word("-Infinity", PyJson::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err("expecting a value".to_string()),
        }
    }

    fn word(&mut self, word: &str, value: PyJson) -> std::result::Result<PyJson, String> {
        self.pos += word.len();
        Ok(value)
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, the longest such prefix. A
    /// literal with a fraction or an exponent is a float; otherwise it is an int.
    fn number(&mut self) -> std::result::Result<PyJson, String> {
        let start = self.pos;
        let digits = |reader: &mut Self| {
            let from = reader.pos;
            while matches!(reader.bytes.get(reader.pos), Some(b'0'..=b'9')) {
                reader.pos += 1;
            }
            reader.pos > from
        };
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err("expecting a digit".to_string()),
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.')
            && matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9'))
        {
            self.pos += 1;
            digits(self);
            is_float = true;
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            let mark = self.pos;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if digits(self) {
                is_float = true;
            } else {
                self.pos = mark;
            }
        }
        let literal = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("0");
        if is_float {
            literal
                .parse::<f64>()
                .map(PyJson::Float)
                .map_err(|e| e.to_string())
        } else {
            Ok(PyJson::Int(literal.to_string()))
        }
    }

    fn hex4(&mut self) -> std::result::Result<u32, String> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .and_then(|b| std::str::from_utf8(b).ok())
            .filter(|s| s.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| "invalid \\u escape".to_string())?;
        self.pos += 4;
        u32::from_str_radix(digits, 16).map_err(|e| e.to_string())
    }

    fn string(&mut self) -> std::result::Result<String, String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let rest = &self.bytes[self.pos..];
            let Some(&byte) = rest.first() else {
                return Err("unterminated string".to_string());
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return Err("unterminated string".to_string());
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let code =
                                if (0xD800..0xDC00).contains(&unit) && self.starts_with("\\u") {
                                    self.pos += 2;
                                    let low = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&low) {
                                        return Err("lone surrogate".to_string());
                                    }
                                    0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                                } else {
                                    unit
                                };
                            out.push(char::from_u32(code).ok_or("lone surrogate")?);
                        }
                        _ => return Err("invalid escape".to_string()),
                    }
                }
                0..=0x1f => return Err("control character in string".to_string()),
                _ => {
                    let text = std::str::from_utf8(rest).map_err(|e| e.to_string())?;
                    let ch = text.chars().next().ok_or("unterminated string")?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn array(&mut self, depth: usize) -> std::result::Result<PyJson, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws()?;
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(PyJson::Array(items));
        }
        loop {
            self.skip_ws()?;
            items.push(self.value(depth + 1)?);
            self.skip_ws()?;
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(PyJson::Array(items));
                }
                _ => return Err("expecting ',' or ']'".to_string()),
            }
        }
    }

    fn object(&mut self, depth: usize) -> std::result::Result<PyJson, String> {
        self.pos += 1;
        let mut entries: Vec<(String, PyJson)> = Vec::new();
        self.skip_ws()?;
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(PyJson::Object(entries));
        }
        loop {
            self.skip_ws()?;
            if self.bytes.get(self.pos) != Some(&b'"') {
                return Err("expecting a property name in double quotes".to_string());
            }
            let key = self.string()?;
            self.skip_ws()?;
            if self.bytes.get(self.pos) != Some(&b':') {
                return Err("expecting ':'".to_string());
            }
            self.pos += 1;
            self.skip_ws()?;
            let value = self.value(depth + 1)?;
            match entries.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key, value)),
            }
            self.skip_ws()?;
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(PyJson::Object(entries));
                }
                _ => return Err("expecting ',' or '}'".to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PyJson;

    // `tokenizer_config.json` is read through this reader: Python's `json.load` accepts the bare
    // constants in unrelated keys (e.g. `model_max_length`), and they are numbers, not strings.
    #[test]
    fn constants_are_floats_and_strings_stay_strings() {
        let value = PyJson::parse(
            r#"{"model_max_length": Infinity, "x": [NaN, -Infinity], "cls_token": "[CLS] NaN",
                "t": "Infinity", "big": 123456789012345678901234567890, "z": -0, "f": 1e400}"#,
        )
        .unwrap();
        assert_eq!(
            Some(&PyJson::Float(f64::INFINITY)),
            value.get("model_max_length")
        );
        assert_eq!(Some(&PyJson::Str("Infinity".into())), value.get("t"));
        assert_eq!(
            Some("[CLS] NaN"),
            value.get("cls_token").and_then(PyJson::as_str)
        );
        assert_eq!(
            Some(&PyJson::Int("123456789012345678901234567890".into())),
            value.get("big")
        );
        assert_eq!(Some(&PyJson::Int("-0".into())), value.get("z"));
        assert_eq!(Some(&PyJson::Float(f64::INFINITY)), value.get("f"));
        let as_value = value.to_value();
        assert!(!as_value["model_max_length"].is_string());
        assert!(as_value["t"].is_string());
    }

    #[test]
    fn duplicate_keys_keep_first_position_and_last_value_as_a_dict_does() {
        let value = PyJson::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(
            PyJson::Object(vec![
                ("a".into(), PyJson::Int("3".into())),
                ("b".into(), PyJson::Int("2".into()))
            ]),
            value
        );
    }

    #[test]
    fn strings_decode_escapes_and_surrogate_pairs() {
        let value = PyJson::parse(r#"["\u00e9\n\ud835\udc00\/"]"#).unwrap();
        assert_eq!(
            PyJson::Array(vec![PyJson::Str("\u{e9}\n\u{1d400}/".into())]),
            value
        );
    }
}
