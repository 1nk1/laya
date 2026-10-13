# laya-go

A pure-Go port of [laya](../README.md). Standard library only: no cgo, no third-party modules.

The foundation (language detection, questions, sequences, the tokenizer) and the predict path
(`infer`, `agent`) are here. The native ONNX session and the router come in later changes.

| Package | What it is |
| --- | --- |
| `lang` | Script and language detection, the port of `laya/lang.py`: `DetectScript`, `ScriptProfile`, `LatinProfileOf`, `GuessLatinLanguage`, `Analyze`, `DetectLanguage`, `IsEnglish`. |
| `schema` | Questions (`Choice`, `ChoiceOf`, `Score`, `Noul`, `NoulWith`, `Questions`) and results (`Answer`, `Prediction`, `Usage`) every other package shares. Questions are refused with laya's own wording and read or written in the Python wire form. |
| `sequence` | `build_sequence`, `build_head`, `state_room`, `window_budget` and `window_batch_cap`: `EncodeState`, `Build`, `StateRoom`, `PlanWindows`, `BatchCap`. |
| `infer` | The seam to a model runtime: `Batch` (the graph's five inputs), `Output` (its two), `Session`, and the shape checks `Batch.Validate` and `Output.Check`. |
| `agent` | `ONNXAgent.predict` and `predict_batch` over an `infer.Session`: `Open`, `Predict`, `PredictBatch`, `LoadConfig`, with batch size, sort-by-length and context cancellation between runs. |
| `tokenizer` | A reader and runner for Hugging Face `tokenizer.json` (BPE): normalizer, pre-tokenizer, added tokens, byte fallback, decoder. |
| `internal/unicodedata` | The character predicates `laya/lang.py` leans on, recorded from CPython. |
| `internal/pyjson` | `json.dumps` and `repr` spelled as CPython spells them. |
| `internal/pyvalue` | `json.loads` with order and number spelling kept, and the Go shapes a state or criterion can take, mapped onto `pyjson`. |
| `internal/decode` | Logit rows to typed answers: `Agent._decode_answers`, its temperatures, binning and `round(x, 4)`. |
| `internal/collate` | Sequences to the five flattened tensors the graph takes. |

## Quickstart

```go
a := lang.Analyze("Ich kann mich nicht in mein Konto einloggen")
fmt.Println(a.Script, a.Language, a.English) // latin de false

lang.IsEnglish(map[string][]string{"msg": {"No puedo entrar en mi cuenta"}}) // false
lang.Analyze("") // English: true, Script: lang.ScriptUnknown
```

A `lang` state is a string, bytes, or any map, slice or array of them, nested to six levels, as
`laya/lang.py` reads one; a struct, a number or nil reads as no text. A Go map has no insertion
order, so its values are read in sorted key order (like `fmt` for one key type; mixed key types by
kind, then type name), and a verdict can differ from a Python dict with the same items when the
4000-character budget truncates. Use `lang.Object`, which decodes JSON in document order, for an
order-sensitive state. `Analyze` reads every leaf, so its cost grows with the number of leaves.

## Layout

- No root package: the module path ends in `laya-go`, so tags are `laya-go/vX.Y.Z`.
- Every package in this module is pure Go: `lang`, `schema`, `sequence`, `tokenizer`, `infer` and
  `agent` now, and `router`, `presets` and `shortlist` later.
- cgo will live only in an `onnx` package, a nested module with its own `go.mod`, so `go test ./...`
  here never compiles it. Importing `lang` or `agent` never compiles cgo.
- `tokenizer` is public (`LoadFS`, `Parse`); `internal/unicodedata`, `internal/pyjson`,
  `internal/decode` and `internal/collate` stay internal. `infer.Batch` is the public name of
  what `collate` builds.

## Parity
## Parity

Python is the source of truth. The tests compare against goldens recorded from it, and the Unicode
behaviour that decides an answer is compiled in from the reference instead of taken from the Go
toolchain, whose Unicode release changes between versions.

```sh
python laya-go/scripts/gen_tables.py            # regenerate the compiled tables
python laya-go/scripts/gen_tables.py --check    # fail if a committed table is stale
python laya-go/scripts/gen_go_fixtures.py       # regenerate laya-go/fixtures/go_extra.json
```

Run them under Python 3.12, the version the lane pins. `go_extra.json` records the interpreter's
Unicode version (15.0.0), and `--check` under an interpreter with another one (3.11 has 14.0.0)
reports the Unicode-dependent families as stale.

Tests read goldens from `$LAYA_GO_FIXTURES` (default: `laya-java/fixtures`, plus the committed
`laya-go/fixtures/go_extra.json`), and the tokenizer tests need `$LAYA_CHECKPOINTS` to point at a
directory holding the `english` and `multilingual` checkpoints. Without it they skip, and the CI
lane fails a run in which anything skipped.

```sh
cd laya-go
LAYA_CHECKPOINTS=/path/to/checkpoints go test ./...
LAYA_CHECKPOINTS=/path/to/checkpoints go test -bench Encode -run '^$' ./tokenizer
```

CI (`.github/workflows/go.yml`) regenerates every golden from Python at the commit under test into
a scratch directory, points the tests at it, and fails on any skip.
