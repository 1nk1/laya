# Changelog

All notable changes to the `Laya.Onnx` NuGet package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-dotnet-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, tested against laya (Python) 0.4.11 (routing
goldens: 0.4.0).

### Added

- `LayaEngine.PredictLong` and `LayaRouter.PredictLong`, ported from Python
  `ONNXAgent.predict_long`: overlapping token windows, batched inference, deciding-window
  attribution, per-type aggregation and summed scan diagnostics.
- `LayaEngine`: `choice`, `score` and `noul` predictions over the exported ONNX checkpoints
  (english, multilingual, typed-decisions); all questions for a state run in one batched forward pass.
- Loads both ONNX layouts: the fused `model.onnx` and the split `encoder.onnx` + `head.onnx`.
- Answers report `confidence` and `answer_confidence` (added in Python 0.3.21).
- Python-compatible sequence building, JSON serialization and calibration, checked against goldens
  recorded from the Python SDK.
- Checkpoint router, language detection, email state and option shortlisting.
- Opt-in model download from Hugging Face.
- Samples: quickstart, routing and benchmark.

### Fixed

- `LayaRouterOptions.MaxLoaded` defaults to 2, as Python's `Router(max_loaded=2)` does, so a
  router that alternates two languages no longer reloads a checkpoint on every request.
- Undetermined Latin text uses the router's configured default, including its route reason.
- Language detection counts unlisted scripts, distinguishes mixed-language fields and lines
  (`MixedSegment`), and applies current script, stopword and loanword rules.
- Email cleaning preserves unpunctuated requests before disclaimers, distinguishes prose from
  signatures and reply headers, and handles Portuguese, Spanish and French client markers.
- Oversized questions report the surviving option markers and the actual sequence budget.
- Score legends return the same rendered criterion text shown to the model, including JSON for
  structured levels.
- Golden fixtures regenerated from current Python for all three checkpoints and routing.

### Known gaps

- Python-only opt-in features are not yet ported: abstention (`min_confidence`),
  `lang_temperatures`, hooks, batching APIs, question option ordering and digest controls.
- `LayaEmail.State` does not expose Python's `max_chars` keyword; use `CleanBody` with a custom
  budget and pass the cleaned body with `clean: false`.
- New Python usage diagnostics (`state_tokens`, `truncated`, `state_tokens_dropped` and
  `truncated_questions`) are populated for `PredictLong`, but not ordinary `Predict`.
