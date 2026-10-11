# laya-go

A pure-Go port of [laya](../README.md). Standard library only: no cgo, no third-party modules.

This first slice is the foundation the rest of the port stands on. The routing and model layers
(ONNX, agent, router) come in later changes.

| Package | What it is |
| --- | --- |
| `lang` | Script and language detection, the port of `laya/lang.py`: `DetectScript`, `ScriptProfile`, `LatinProfileOf`, `GuessLatinLanguage`, `Analyze`, `DetectLanguage`, `IsEnglish`. |
| `internal/tokenizer` | A reader and runner for Hugging Face `tokenizer.json` (BPE): normalizer, pre-tokenizer, added tokens, byte fallback, decoder. |
| `internal/unicodedata` | The character predicates `laya/lang.py` leans on, recorded from CPython. |
| `internal/pyjson` | `json.dumps` and `repr` spelled as CPython spells them. |

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
- Pure-Go packages are `lang` now and `router`, `presets` and `shortlist` later.
- cgo, if ever needed, lives only in a future `onnx` package, with a `//go:build !cgo` stub.
  Importing `lang` never compiles cgo.
- The tokenizer goes public in the next change; `internal/unicodedata` and `internal/pyjson`
  stay internal.

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

Tests read goldens from `$LAYA_GO_FIXTURES` (default: `laya-java/fixtures`, plus the committed
`laya-go/fixtures/go_extra.json`), and the tokenizer tests need `$LAYA_CHECKPOINTS` to point at a
directory holding the `english` and `multilingual` checkpoints. Without it they skip, and the CI
lane fails a run in which anything skipped.

```sh
cd laya-go
LAYA_CHECKPOINTS=/path/to/checkpoints go test ./...
LAYA_CHECKPOINTS=/path/to/checkpoints go test -bench Encode -run '^$' ./internal/tokenizer
```

CI (`.github/workflows/go.yml`) regenerates every golden from Python at the commit under test into
a scratch directory, points the tests at it, and fails on any skip.
