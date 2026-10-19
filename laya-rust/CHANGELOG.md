# Changelog

All notable changes to the `laya-onnx` crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-rust-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, checked against goldens and routing probes
regenerated from laya (Python) 0.4.11 for the supported paths (see the known gaps below).

### Added

- `LangGuess` and `LayaRouter::{route_with_lang_guess, predict_with_lang_guess,
  predict_long_with_lang_guess, with_lang_guess}`: Python's `lang_guess` hint (a language code or a
  callback taking the state), per call or installed on the router, checked after an explicit `lang`
  and before detection. A hint that names no language falls through to detection.
- `LanguageAnalysis::mixed_segment`, the line or field the segment scan singled out in a state that
  reads English as a whole, and the matching routing reason ("Latin script, mostly English, but a
  line or field reads as ..."). `LanguageAnalysis` is now `#[non_exhaustive]`, so a field added
  later does not break callers again; read it, do not construct it.
- `LayaEmail::{clean_body_signed, state_with_max_chars}`: `max_chars` as Python takes it, including
  the `max_chars * 4` pre-cut and the specified results for `0` and negative budgets. `clean_body`
  and `state` are unchanged and keep the 3000-character default.
- `LayaEngine::predict_long` and `LayaRouter::predict_long` (`aggregate="auto"`): scan a state
  longer than one sequence in overlapping token windows and keep the most confident window per
  question, with `Answer::window()` attribution, `LayaResult::scan_usage()` and summed
  `state_usage()`. Matches Python's `ONNXAgent.predict_long` and the .NET SDK; hooks, per-language
  temperatures and the confidence gate are not part of it. New `LongOptions`, `LayaLongPredictor`,
  `LayaError::{UnsupportedAggregate, InvalidWindow, NotSupported}` and a default
  `LayaTokenizer::decode`. `LayaError` is not `#[non_exhaustive]`, so an exhaustive `match` on it
  outside the crate needs arms for the three new variants.
- `predict` now runs through a shared multi-state core that scores several states in one forward
  pass; its results are unchanged.
- Advisory parity CI on every push and PR: regenerate fixtures from the current Python checkout,
  test all checkpoints, and reject skipped tests. No golden cache or crate publication.
- Python float-repr parity probes, including exact half-even decimal ties and random bit patterns.
- `LayaResult::state_usage()` reports state-token truncation across question budgets, matching
  Python 0.4.0 without changing the existing `Usage` totals or result constructor.
- Engine score legends render structured criteria as strings, matching Python 0.4.0.
- `NoMarkers` errors include actual/requested marker counts and the total sequence budget,
  matching Python's actionable truncation diagnostic.
- State usage also reports option-span collisions after clipping, including the applied cap.
- `LayaEngine`: `choice`, `score` and `noul` predictions over the exported ONNX checkpoints
  (english, multilingual, typed-decisions); all questions for a state run in one batched forward pass.
- Loads both ONNX layouts: the fused `model.onnx` and the split `encoder.onnx` + `head.onnx`.
- Answers report `confidence` and `answer_confidence` (added in Python 0.3.21).
- Python-compatible sequence building, JSON serialization and calibration, checked against goldens
  recorded from the Python SDK.
- Checkpoint router, language detection, email state and option shortlisting.
- Opt-in model download from Hugging Face (Cargo feature `download`).
- Samples: quickstart, routing and benchmark.

### Fixed

- `LayaRouterOptions::max_loaded` defaults to 2, as Python's `Router(max_loaded=2)` does, so a
  router that alternates two languages no longer reloads a checkpoint on every request.
- Language detection follows Python 0.4.11: the line-by-line and field-by-field scan of a state that
  reads English as a whole, the Swedish and Nordic rules (Swedish, Banglish and Azerbaijani word
  lists, short Swedish requests, the Swedish-Danish overlap marker), the English rescue of a single
  accented loanword, acronym-shaped and all-caps lines no longer named foreign, identifier and code
  lines skipped, fullwidth and IPA letters counted as Latin (cutoff U+02B0) and letters no script
  range claims counted as "other". Python's `\w`, `\s`, `isupper` and `unicodedata.combining` are
  reproduced character for character rather than through the `regex` crate.
- A `lang` of `C`, `POSIX`, `C.UTF-8`, `und`, `zxx`, `mul` or a blank string no longer pins a
  checkpoint: it falls through to `lang_guess` and detection. `en_US` and `en_US.UTF-8` now read as
  English instead of routing to the multilingual checkpoint.
- Routing reasons quote caller text like Python's `%r`, escaping non-printable characters.
- Email cleaning follows Python 0.4.11: nine quote-header patterns (Portuguese, Spanish and French
  attributions that carry a date, Outlook headers in four languages, bare `From:`/`De:` headers
  confirmed by the `Sent:`/`Enviado:`/`Date:` line after them), the wrapped-attribution tail,
  Portuguese/Spanish/French sign-offs, the English sign-off name rule, device and mail-app footers,
  and Portuguese/Spanish/French disclaimers. A bare `From:` line that opens a sentence is kept.
- Dependency: `unicode-normalization` (canonical combining classes for `unicodedata.combining`).

### Known gaps

- Language detection, routing and email cleaning follow the regenerated Python 0.4.11 probes; a
  later Python change needs the probes regenerated.
- Not yet ported: abstention (`min_confidence`, including per-bucket thresholds),
  histogram-binning calibration and `lang_temperatures`.
