//! `float(str)` for string temperatures, as Python's `clamp_temperature` reads them. The applied
//! values and the warning text were produced by Python's `clamp_temperature`,
//! `temperature_rejected` and the loader's format string on the same 37 strings.

use laya::laya_config::LayaConfig;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());
static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

const CONFIG: &str = "{\"temperature\": [\"2.5\", \" 2.5 \", \"\\t2.5\\n\", \"+2.5\", \"-1\", \"1e1\", \"1E-1\", \".5\", \"5.\", \"2_5\", \"1_000\", \"_1\", \"1_\", \"1__0\", \"1_.5\", \"1e_1\", \"0x1\", \"abc\", \"\", \"nan\", \"-inf\", \"Infinity\", \"+INF\", \"\\uff12.\\uff15\", \"\\u0662\", \"\\u00a02\", \"\\u30002.5\\u3000\", \"\\u00852\", \"1e500\", \"1_0e1_0\", \"\\ud835\\udfd0\", \".\", \"e5\", \"1e\", \"1.5.2\", \"--1\", \"2 5\"]}";
const APPLIED: [f64; 37] = [
    2.5, 2.5, 2.5, 2.5, 0.5, 5.0, 0.5, 0.5, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 2.5, 2.0, 2.0, 2.5, 2.0, 1.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const WARNING: &str = "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping temperature[4]='-1' -> 0.5, temperature[5]='1e1' -> 5, temperature[6]='1E-1' -> 0.5, temperature[9]='2_5' -> 5, temperature[10]='1_000' -> 5, temperature[11]='_1' -> 1, temperature[12]='1_' -> 1, temperature[13]='1__0' -> 1, temperature[14]='1_.5' -> 1, temperature[15]='1e_1' -> 1, temperature[16]='0x1' -> 1, temperature[17]='abc' -> 1, temperature[18]='' -> 1, temperature[19]='nan' -> 1, temperature[20]='-inf' -> 1, temperature[21]='Infinity' -> 1, temperature[22]='+INF' -> 1, temperature[28]='1e500' -> 1, temperature[29]='1_0e1_0' -> 5, temperature[31]='.' -> 1, temperature[32]='e5' -> 1, temperature[33]='1e' -> 1, temperature[34]='1.5.2' -> 1, temperature[35]='--1' -> 1, temperature[36]='2 5' -> 1. Treat confidence from the affected buckets as uncalibrated.";

#[test]
fn string_temperatures_follow_python_float_parsing() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();

    let config = LayaConfig::parse(CONFIG).unwrap();
    assert_eq!(APPLIED.to_vec(), config.temperature);
    let seen = SEEN.lock().unwrap();
    assert_eq!(vec![WARNING.to_string()], *seen);
}

// Python: json.loads gives an `int` for a 400-digit literal and clamp_temperature raises
// OverflowError("int too large to convert to float") uncaught; `1e400` is a float (inf -> 1.0);
// 10**308 is still a valid int (-> 5.0); 10**309 overflows.
#[test]
fn huge_integer_temperatures_fail_to_load_as_in_python_and_float_literals_do_not() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();
    let list = |entry: &str| format!(r#"{{"temperature": [1.0, {entry}, 1.0]}}"#);
    let by_options =
        |entry: &str| format!(r#"{{"temperature_by_options": {{"choice:2": {entry}}}}}"#);
    let huge = format!("1{}", "0".repeat(400));
    for config in [
        list(&huge),
        list(&format!("-{huge}")),
        list(&format!("1{}", "0".repeat(309))),
        by_options(&huge),
    ] {
        let error = LayaConfig::parse(&config).unwrap_err();
        assert!(
            matches!(error, laya::error::LayaError::InvalidConfig(_)),
            "{error:?}"
        );
        assert_eq!(
            "OverflowError: int too large to convert to float",
            error.to_string()
        );
    }
    for (entry, applied) in [
        ("1e400".to_string(), 1.0),
        ("-1e400".to_string(), 1.0),
        (format!("1.{}", "0".repeat(400)), 1.0),
        (format!("1{}", "0".repeat(308)), 5.0),
    ] {
        let config = LayaConfig::parse(&list(&entry)).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(vec![1.0, applied, 1.0], config.temperature, "{entry}");
    }
    // Python: temperature[1]=inf -> 1 (a float, so repr is `inf`, not a string)
    let seen = SEEN.lock().unwrap();
    assert!(
        seen[0].contains("clamping temperature[1]=inf -> 1. Treat"),
        "{}",
        seen[0]
    );
    assert!(
        seen[1].contains("clamping temperature[1]=-inf -> 1. Treat"),
        "{}",
        seen[1]
    );
}
