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
use crate::python_json_reader::PyJson;
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
fn temperature_number(raw: &PyJson) -> Option<f64> {
    match raw {
        PyJson::Int(literal) => PyJson::int_as_f64(literal),
        PyJson::Float(f) => Some(*f),
        PyJson::Str(s) => py_float(s),
        _ => None,
    }
}

/// Python's `OverflowError` from `clamp_temperature(<huge int>)`, which nothing catches, so the
/// checkpoint fails to load.
fn check_temperature_overflow(root: &PyJson) -> Result<()> {
    let overflows = |v: &PyJson| matches!(v, PyJson::Int(l) if PyJson::int_as_f64(l).is_none());
    let by_options = match root.get("temperature_by_options") {
        Some(PyJson::Object(entries)) => entries.iter().any(|(_, v)| overflows(v)),
        _ => false,
    };
    let list = match root.get("temperature") {
        Some(PyJson::Array(items)) => items.iter().any(overflows),
        _ => false,
    };
    if by_options || list {
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
fn py_repr(value: &PyJson) -> String {
    match value {
        PyJson::Null => "None".to_string(),
        PyJson::Bool(true) => "True".to_string(),
        PyJson::Bool(false) => "False".to_string(),
        PyJson::Int(literal) => match literal.strip_prefix('-') {
            // Python's `int('-0')` is `0`.
            Some("0") => "0".to_string(),
            _ => literal.clone(),
        },
        PyJson::Float(f) if f.is_nan() => "nan".to_string(),
        PyJson::Float(f) if f.is_infinite() => if *f > 0.0 { "inf" } else { "-inf" }.to_string(),
        PyJson::Float(f) => PythonJson::repr(*f),
        PyJson::Str(s) => python_repr_str(s),
        PyJson::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        PyJson::Object(map) => format!(
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
fn temperature_warning(root: &PyJson) -> Option<String> {
    let mut entries: Vec<(String, &PyJson)> = Vec::new();
    if let Some(PyJson::Object(by_options)) = root.get("temperature_by_options") {
        entries.extend(by_options.iter().map(|(k, v)| (k.clone(), v)));
    }
    if let Some(PyJson::Array(list)) = root.get("temperature") {
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
    #[track_caller]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse a config from JSON text.
    #[track_caller]
    pub fn parse(json: &str) -> Result<Self> {
        let at = std::panic::Location::caller();
        let root = PyJson::parse(json)?;
        check_temperature_overflow(&root)?;

        let mut temperature_raw: Vec<f64> = match root.get("temperature") {
            Some(PyJson::Array(arr)) => arr
                .iter()
                .map(|v| temperature_number(v).unwrap_or(f64::NAN))
                .collect(),
            _ => Vec::new(),
        };
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while temperature_raw.len() < 3 {
            temperature_raw.push(1.0);
        }

        let temperature_by_options_raw: HashMap<String, f64> =
            match root.get("temperature_by_options") {
                Some(PyJson::Object(obj)) => obj
                    .iter()
                    .map(|(k, v)| (k.clone(), temperature_number(v).unwrap_or(f64::NAN)))
                    .collect(),
                _ => HashMap::new(),
            };

        let encoder = root
            .get("encoder")
            .and_then(PyJson::as_str)
            .map(str::to_string);
        let max_len = root
            .get("max_len")
            .and_then(PyJson::as_u64)
            .map(|v| v as usize)
            .unwrap_or(512);
        let head_max_len = root
            .get("head_max_len")
            .and_then(PyJson::as_u64)
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
            emit_warnings(&[warning], at);
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
