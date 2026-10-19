//! The contents of a checkpoint's `rl_agent_config.json`. Ports the config shape `laya/agent.py`'s
//! `Agent.system_one` reads via `cfg.get(...)`.
//!
//! Defaults match Python's `cfg.get(...)` fallbacks in `Agent.system_one`, not the shipped
//! multilingual values, so a config missing a key behaves the same in both runtimes.

use crate::calibration::Calibration;
use crate::error::{LayaError, Result};
use crate::language_detection::py_strip;
use crate::laya_router::python_repr_str;
use crate::long_prediction::emit_warnings;
use crate::python_json::PythonJson;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use unicode_general_category::{GeneralCategory, get_general_category};

/// The parsed and clamped contents of `rl_agent_config.json`.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    /// The encoder the checkpoint was trained on, for information only.
    pub encoder: Option<String>,
    /// Maximum total sequence length.
    pub max_len: usize,
    /// Token budget for the question head: instructions plus all option fragments.
    pub head_max_len: usize,
    /// Per-question-type temperature, clamped to the usable range. Always 3 entries.
    pub temperature: Vec<f64>,
    /// Per `(type, option-count)` bucket temperature, clamped. Keys as
    /// [`Calibration::temp_bucket`] builds them.
    pub temperature_by_options: HashMap<String, f64>,
    /// What the checkpoint shipped, before clamping.
    pub temperature_raw: Vec<f64>,
    /// What the checkpoint shipped, before clamping.
    pub temperature_by_options_raw: HashMap<String, f64>,
    /// Descriptions of every temperature that had to be clamped, empty when none were. A
    /// non-empty list means confidence from those buckets is uncalibrated. [`Self::parse`] also
    /// sends Python's warning for it to the `log` facade (`warn`, target `laya`).
    pub clamped_temperatures: Vec<String>,
}

/// The value of a Unicode decimal digit (`Nd`), as Python's `int()`/`float()` read it. The `Nd`
/// characters come in runs of ten starting at zero, and runs can touch (the mathematical digits),
/// so the value is the distance from the start of the unbroken `Nd` stretch, modulo ten.
fn decimal_digit_value(ch: char) -> Option<u32> {
    let is_nd = |c: char| get_general_category(c) == GeneralCategory::DecimalNumber;
    if !is_nd(ch) {
        return None;
    }
    let mut back = 0u32;
    let mut code = ch as u32;
    while code > 0 && char::from_u32(code - 1).is_some_and(is_nd) {
        back += 1;
        code -= 1;
    }
    Some(back % 10)
}

/// Python's `float(str)`: surrounding whitespace is stripped, then an optional sign and either
/// `inf`/`infinity`/`nan` (any case) or decimal digits (any script) with at most one `_` between
/// two digits, an optional `.` and fraction, and an optional exponent. Anything else is `None`
/// (Python's `ValueError`).
pub(crate) fn py_float(text: &str) -> Option<f64> {
    let text = py_strip(text);
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let lower = body.to_ascii_lowercase();
    let magnitude = match lower.as_str() {
        "inf" | "infinity" => f64::INFINITY,
        "nan" => f64::NAN,
        _ => {
            let ascii = ascii_float_text(body)?;
            ascii.parse::<f64>().ok()?
        }
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// `digitpart ["." [digitpart]] | "." digitpart`, then `[eE] [sign] digitpart`, rewritten with
/// ASCII digits and no underscores; `None` when `body` does not have that shape.
fn ascii_float_text(body: &str) -> Option<String> {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    // A run of digits with single underscores between digits; returns how many digits it read.
    let digits = |i: &mut usize, out: &mut String| -> usize {
        let mut count = 0;
        while *i < chars.len() {
            if let Some(d) = decimal_digit_value(chars[*i]) {
                out.push(char::from_digit(d, 10).unwrap());
                count += 1;
                *i += 1;
            } else if chars[*i] == '_'
                && count > 0
                && chars
                    .get(*i + 1)
                    .is_some_and(|c| decimal_digit_value(*c).is_some())
            {
                *i += 1;
            } else {
                break;
            }
        }
        count
    };
    let whole = digits(&mut i, &mut out);
    let mut fraction = 0;
    if chars.get(i) == Some(&'.') {
        out.push('.');
        i += 1;
        fraction = digits(&mut i, &mut out);
    }
    if whole + fraction == 0 {
        return None;
    }
    if matches!(chars.get(i), Some('e' | 'E')) {
        out.push('e');
        i += 1;
        if let Some(&sign @ ('+' | '-')) = chars.get(i) {
            out.push(sign);
            i += 1;
        }
        if digits(&mut i, &mut out) == 0 {
            return None;
        }
    }
    (i == chars.len()).then_some(out)
}

/// A JSON temperature entry as `clamp_temperature` reads it: a number as itself, a string through
/// `float(str)`, and a bool or anything else not at all.
fn temperature_number(raw: &Value) -> Option<f64> {
    match raw {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s == NAN_MARK => Some(f64::NAN),
        Value::String(s) if s == POS_INF_MARK => Some(f64::INFINITY),
        Value::String(s) if s == NEG_INF_MARK => Some(f64::NEG_INFINITY),
        Value::String(s) => py_float(s),
        _ => None,
    }
}

/// Stands in, in the re-read text, for a JSON integer literal too large for a float.
/// Stands in for a float literal that overflows to infinity (Python: `inf`, `-inf`).
const POS_INF_MARK: &str = "__laya_inf__";
const NEG_INF_MARK: &str = "__laya_neg_inf__";

const INT_OVERFLOW_MARK: &str = "__laya_int_too_large_to_convert_to_float__";

/// Whether `token` has the shape of a JSON number: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
fn is_json_number(token: &str) -> bool {
    let b = token.as_bytes();
    let digits = |mut i: usize| {
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        (i, i > start)
    };
    let mut i = usize::from(b.first() == Some(&b'-'));
    if b.get(i) == Some(&b'0') {
        i += 1;
    } else {
        let (next, any) = digits(i);
        if !any || b[i] == b'0' {
            return false;
        }
        i = next;
    }
    if b.get(i) == Some(&b'.') {
        let (next, any) = digits(i + 1);
        if !any {
            return false;
        }
        i = next;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let (next, any) = digits(i);
        if !any {
            return false;
        }
        i = next;
    }
    i == b.len()
}

/// Stands in for the bare token `NaN`, which Python's `json.loads` accepts as `float('nan')`.
const NAN_MARK: &str = "__laya_nan__";

/// The bare constants Python's `json.loads` accepts and serde_json refuses (`NaN`, `Infinity`,
/// `-Infinity`; nothing else, so `nan`, `+Infinity` and `Inf` stay errors) at the start of `rest`:
/// the marker standing in for it and its length in characters.
fn python_json_constant(rest: &[char]) -> Option<(&'static str, usize)> {
    let starts = |word: &str| {
        let n = word.chars().count();
        rest.len() >= n
            && rest[..n].iter().copied().eq(word.chars())
            && !rest
                .get(n)
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
    };
    if starts("NaN") {
        Some((NAN_MARK, 3))
    } else if starts("Infinity") {
        Some((POS_INF_MARK, 8))
    } else if starts("-Infinity") {
        Some((NEG_INF_MARK, 9))
    } else {
        None
    }
}

/// `json.loads`, for the places serde_json is stricter than Python:
/// - it refuses a number literal that overflows an `f64`, where Python keeps a huge integer as an
///   `int` and reads `1e400` as `inf`;
/// - it refuses the bare tokens `NaN`, `Infinity` and `-Infinity`, which Python reads as floats.
///
/// Such a literal is rewritten before a second read: a float literal or constant becomes a marker
/// string read back as the float Python makes of it, an integer literal that overflows becomes
/// [`INT_OVERFLOW_MARK`]. Text serde_json reads as is never reaches the rewrite.
pub(crate) fn read_json(text: &str) -> Result<Value> {
    let first = match serde_json::from_str::<Value>(text) {
        Ok(value) => return Ok(value),
        Err(e) => e,
    };
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let (mut i, mut in_string, mut escaped) = (0, false, false);
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
        } else if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
        } else if let Some((mark, len)) = python_json_constant(&chars[i..]) {
            out.push_str(&format!("\"{mark}\""));
            i += len;
        } else if c == '-' || c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && matches!(chars[i], '-' | '+' | '.' | 'e' | 'E' | '0'..='9') {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            if !is_json_number(&token) || serde_json::from_str::<Value>(&token).is_ok() {
                out.push_str(&token);
            } else if token.contains(['.', 'e', 'E']) {
                let mark = if token.starts_with('-') {
                    NEG_INF_MARK
                } else {
                    POS_INF_MARK
                };
                out.push_str(&format!("\"{mark}\""));
            } else {
                out.push_str(&format!("\"{INT_OVERFLOW_MARK}\""));
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    serde_json::from_str(&out).map_err(|_| first.into())
}

/// Python's `OverflowError` from `clamp_temperature(<huge int>)`, which nothing catches, so the
/// checkpoint fails to load.
fn check_temperature_overflow(root: &Value) -> Result<()> {
    let marked = |v: &Value| v.as_str() == Some(INT_OVERFLOW_MARK);
    let by_options = root
        .get("temperature_by_options")
        .and_then(Value::as_object);
    let list = root.get("temperature").and_then(Value::as_array);
    if by_options.is_some_and(|m| m.values().any(marked))
        || list.is_some_and(|l| l.iter().any(marked))
    {
        return Err(LayaError::InvalidConfig(
            "OverflowError: int too large to convert to float".to_string(),
        ));
    }
    Ok(())
}

/// `%g` of a clamped temperature (six significant digits, trailing zeros dropped); the value is
/// always inside `[0.5, 5]` or the neutral `1`, so no exponent form is needed.
fn percent_g(value: f64) -> String {
    let exponent: i32 = format!("{value:.5e}")
        .split('e')
        .nth(1)
        .and_then(|e| e.parse().ok())
        .unwrap_or(0);
    let decimals = (5 - exponent).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// Python's `repr()` of a JSON value as `json.load` would have produced it.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => PythonJson::repr(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) if s == NAN_MARK => "nan".to_string(),
        Value::String(s) if s == POS_INF_MARK => "inf".to_string(),
        Value::String(s) if s == NEG_INF_MARK => "-inf".to_string(),
        Value::String(s) => python_repr_str(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", python_repr_str(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The text of the `warnings.warn` in `ONNXAgent.__init__` for a checkpoint whose temperatures
/// had to be replaced, or `None` when every one was accepted. Entries are named, in Python's
/// order (`temperature_by_options` in file order, then `temperature[i]`), with the value as
/// shipped (`%r`) and the one applied (`%g`).
fn temperature_warning(root: &Value) -> Option<String> {
    let mut entries: Vec<(String, &Value)> = Vec::new();
    if let Some(by_options) = root
        .get("temperature_by_options")
        .and_then(Value::as_object)
    {
        entries.extend(by_options.iter().map(|(k, v)| (k.clone(), v)));
    }
    if let Some(list) = root.get("temperature").and_then(Value::as_array) {
        entries.extend(
            list.iter()
                .enumerate()
                .map(|(i, v)| (format!("temperature[{i}]"), v)),
        );
    }
    let rejected: Vec<String> = entries
        .into_iter()
        .filter_map(|(name, raw)| {
            // A bool or anything that is not a number is replaced by the neutral 1.0.
            let number = temperature_number(raw);
            let applied = Calibration::clamp_temperature(number.unwrap_or(f64::NAN));
            (number != Some(applied))
                .then(|| format!("{name}={} -> {}", py_repr(raw), percent_g(applied)))
        })
        .collect();
    if rejected.is_empty() {
        return None;
    }
    Some(format!(
        "laya ONNX: this checkpoint ships temperatures outside [{}, {}] which would distort \
         confidence; clamping {}. Treat confidence from the affected buckets as uncalibrated.",
        percent_g(Calibration::TEMP_MIN),
        percent_g(Calibration::TEMP_MAX),
        rejected.join(", ")
    ))
}

impl LayaConfig {
    /// Read a config from a `rl_agent_config.json` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse a config from JSON text.
    pub fn parse(json: &str) -> Result<Self> {
        let root = read_json(json)?;
        check_temperature_overflow(&root)?;

        let mut temperature_raw: Vec<f64> = root
            .get("temperature")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|v| temperature_number(v).unwrap_or(f64::NAN))
                    .collect()
            })
            .unwrap_or_default();
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while temperature_raw.len() < 3 {
            temperature_raw.push(1.0);
        }

        let temperature_by_options_raw: HashMap<String, f64> = root
            .get("temperature_by_options")
            .and_then(Value::as_object)
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), temperature_number(v).unwrap_or(f64::NAN)))
                    .collect()
            })
            .unwrap_or_default();

        let encoder = root
            .get("encoder")
            .and_then(Value::as_str)
            .map(str::to_string);
        let max_len = root
            .get("max_len")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(512);
        let head_max_len = root
            .get("head_max_len")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(192);

        let temperature: Vec<f64> = temperature_raw
            .iter()
            .copied()
            .map(Calibration::clamp_temperature)
            .collect();
        let temperature_by_options: HashMap<String, f64> = temperature_by_options_raw
            .iter()
            .map(|(k, &v)| (k.clone(), Calibration::clamp_temperature(v)))
            .collect();

        let mut clamped_temperatures = Vec::new();
        for (bucket, &raw) in &temperature_by_options_raw {
            if Calibration::clamp_temperature(raw) != raw {
                clamped_temperatures.push(format!("{bucket}={raw}"));
            }
        }
        for (i, &raw) in temperature_raw.iter().enumerate() {
            if Calibration::clamp_temperature(raw) != raw {
                clamped_temperatures.push(format!("temperature[{i}]={raw}"));
            }
        }

        if let Some(warning) = temperature_warning(&root) {
            emit_warnings(&[warning]);
        }

        Ok(LayaConfig {
            encoder,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            temperature_raw,
            temperature_by_options_raw,
            clamped_temperatures,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::read_json;

    // `tokenizer_config.json` goes through the same reader: Python's `json.load` accepts these
    // constants in unrelated keys (e.g. `model_max_length`), so the special-token read must too.
    #[test]
    fn read_json_accepts_pythons_constants_and_keeps_strings_alone() {
        let value = read_json(
            r#"{"model_max_length": Infinity, "x": [NaN, -Infinity], "cls_token": "[CLS] NaN"}"#,
        )
        .unwrap();
        assert_eq!("[CLS] NaN", value["cls_token"].as_str().unwrap());
        assert!(value["x"].is_array() && value["model_max_length"].is_string());
        assert!(read_json(r#"{"a": nan}"#).is_err());
    }
}
