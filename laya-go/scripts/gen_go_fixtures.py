#!/usr/bin/env python3
"""Record the goldens laya-go needs that laya-java's fixtures do not hold: go_extra.json.

laya-java/scripts/gen_fixtures.py records what Python answers for the tokenizer corpus, the
decode corpus, language detection and python_json. That is enough to prove the Java port on the
inputs somebody thought of. This port also re-implements what the JVM supplied for free --
Unicode normalization, lowercasing, the pre-tokenizer's scanner -- so it is held to the
reference over far more input than a hand-written corpus, from a fixed seed:

  * normalize       the crate's NFC/NFD/NFKC/NFKD over every code point (a digest), and over
                    random strings of marks, jamo and compatibility characters
  * lowercase       the crate's Lowercase over every code point (a digest)
  * python_lower    CPython's str.lower() on strings built around a capital sigma, the one
                    context-dependent rule
  * single_word     which code points the crate counts as word characters beside a
                    single_word added token (two digests, one per side)
  * pretokenize     ByteLevel (three configurations) and Metaspace (six) on random strings
  * normalizers     Strip, Prepend, Replace (string and regex), Lowercase, Sequence
  * encode_fuzz     ids for random strings on both shipped checkpoints, reaching the heap merge
                    path, added tokens, byte fallback and the unknown token
  * decode_fuzz     decode of random id lists, which produce the invalid UTF-8 a real decode of
                    a truncated slice does

    python laya-go/scripts/gen_go_fixtures.py --out DIR      # needs LAYA_FIXTURE_CHECKPOINTS
    python laya-go/scripts/gen_go_fixtures.py --check        # fail if laya-go/fixtures differs

A checkpoint that cannot be found is an error, never a skip marker.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
GO = os.path.normpath(os.path.join(HERE, ".."))
DEFAULT_OUT = os.path.join(GO, "fixtures")
NAME = "go_extra.json"
MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)
SEED = 20261011


def digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(row.encode("utf-8"))
    return h.hexdigest()


def mapped(cp, text):
    return "%d:%s;" % (cp, ",".join(str(ord(c)) for c in text))


# --- normalization, lowercase, sigma, single_word -------------------------------------------
def normalize_family(rng):
    from tokenizers import normalizers

    forms = {"NFC": normalizers.NFC(), "NFD": normalizers.NFD(),
             "NFKC": normalizers.NFKC(), "NFKD": normalizers.NFKD()}
    sweep = {}
    for name, norm in forms.items():
        sweep[name] = digest(mapped(cp, norm.normalize_str(chr(cp)))
                             for cp in range(MAX) if cp not in SURROGATES)
    marks = [cp for cp in range(0x300, 0x2000) if unicodedata.combining(chr(cp))][:400]
    marks += [0x0345, 0x1DFA, 0x0C3C, 0x07FD, 0x302A, 0x3099, 0x309A, 0x1E4EE, 0x10D69]
    compat = [cp for cp in range(0xA0, 0x1F000)
              if cp not in SURROGATES and unicodedata.normalize("NFKD", chr(cp)) != chr(cp)]
    pools = [
        [ord(c) for c in "aeiouAEIOUnNcCzZ oO"],
        [ord(c) for c in "éèêëÀÁÅçñüÿǆǅ"
         "ﬁﬂ①½ｱｶ㌀㈱℃ÅΩᾳǖṩ"],
        marks, marks,
        list(range(0x1100, 0x1113)), list(range(0x1161, 0x1176)), list(range(0x11A8, 0x11C3)),
        [rng.randrange(0xAC00, 0xD7A4) for _ in range(40)],
        compat,
        [0x200D, 0x200B, 0x20, 0x0A],
    ]
    cases = []
    for _ in range(1500):
        text = "".join(chr(rng.choice(rng.choice(pools))) for _ in range(rng.randrange(1, 9)))
        cases.append({"text": text, **{k.lower(): n.normalize_str(text) for k, n in forms.items()}})
    return {"sweep": sweep, "cases": cases}


def lowercase_family(rng):
    from tokenizers import normalizers

    low = normalizers.Lowercase()
    return {"sweep": digest(mapped(cp, low.normalize_str(chr(cp)))
                            for cp in range(MAX) if cp not in SURROGATES)}


def sigma_family(rng):
    sigma = "Σ"
    pool = ["A", "b", "α", "Α", "Β", sigma, sigma, sigma, " ", "'", ".", "́",
            "‍", "1", "-", "·", "ʰ", "’", "ǅ", "ẞ", "İ", "ͅ",
            "あ", "﻿", ":"]
    cases = []
    for _ in range(1000):
        text = "".join(rng.choice(pool) for _ in range(rng.randrange(1, 8)))
        cases.append({"text": text, "lower": text.lower()})
    cases += [{"text": t, "lower": t.lower()} for t in
              ("ΟΔΣ", "Σ", "AΣ", "ΣA", "AΣB", "ΑΣ Β",
               "A.Σ", "AΣ.", "A'Σ'B", "ΣΣ", "AΣΣ", "ΧΑΟΣ ΣΟΦΟΣ")]
    return {"cases": cases}


def single_word_family(rng):
    from tokenizers import AddedToken, Tokenizer, models

    tok = Tokenizer(models.BPE(vocab={"a": 0}, merges=[], unk_token=None))
    tok.add_tokens([AddedToken("TOK", single_word=True)])
    tid = tok.token_to_id("TOK")
    left, right = [], []
    for cp in range(MAX):
        ch = chr(cp)
        try:
            left.append("1" if tid in tok.encode(ch + "TOK").ids else "0")
            right.append("1" if tid in tok.encode("TOK" + ch).ids else "0")
        except Exception:
            left.append("0")
            right.append("0")
    return {"token": "TOK", "id": tid,
            "left_standalone": digest(left), "right_standalone": digest(right)}


# --- pre-tokenizers ---------------------------------------------------------------------------
TEXT_POOL = (
    ["a", "b", "Hello", "world", "x", "Z", "123", "4", "7.5", "'s", "'t", "'re", "'ve", "'m", "'ll",
     "'d", "'S", "don't", "I'm", "it's", "’s", ",", ".", "!", "?", "-", "_", "/", "(", ")", "$",
     "@", "#", "é", "é", "中", "文", "あ", "한", "م", "א",
     "न", "ก", "Ж", "Ω", "\U0001f600", "‍", "​", "﻿", "½",
     "²", "Ⅷ", "٣", "\U0001d7d8", "\U00010d4a", "\U0002ebf0", "▁", "▁▁"]
    + [" "] * 8 + ["  ", "   ", "\t", "\n", "\r\n", " ", "　", " ", " ", "\u0085"]
)


def fuzz_text(rng, low=1, high=12):
    return "".join(rng.choice(TEXT_POOL) for _ in range(rng.randrange(low, high)))


def pretokenize_family(rng):
    from tokenizers import pre_tokenizers

    texts = ["", " ", "a", "  ", "a  b\t\tc\n\nd\r\ne   f", "abc123", "123abc", "a 1 b", "'s", " 's",
             "x's value", "  x", "x  ", "▁", "▁a▁b", "a▁b", " ▁"]
    texts += [fuzz_text(rng) for _ in range(1200)]
    out = {"texts": texts}
    byte_level = {}
    for prefix in (False, True):
        for use_regex in (True, False):
            pre = pre_tokenizers.ByteLevel(add_prefix_space=prefix, use_regex=use_regex)
            key = "add_prefix_space=%s,use_regex=%s" % (prefix, use_regex)
            byte_level[key] = [[p for p, _ in pre.pre_tokenize_str(t)] for t in texts]
    out["byte_level"] = byte_level
    meta = {}
    for scheme in ("always", "first", "never"):
        for split in (True, False):
            pre = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=scheme, split=split)
            meta["%s,split=%s" % (scheme, split)] = [[p for p, _ in pre.pre_tokenize_str(t)]
                                                     for t in texts]
    out["metaspace"] = meta
    return out


def normalizers_family(rng):
    from tokenizers import Regex, normalizers

    def seq(*items):
        return normalizers.Sequence(list(items))

    specs = [
        ({"type": "Lowercase"}, normalizers.Lowercase()),
        ({"type": "Strip", "strip_left": True, "strip_right": True}, normalizers.Strip(True, True)),
        ({"type": "Strip", "strip_left": True, "strip_right": False}, normalizers.Strip(True, False)),
        ({"type": "Strip", "strip_left": False, "strip_right": True}, normalizers.Strip(False, True)),
        ({"type": "Prepend", "prepend": "▁"}, normalizers.Prepend("▁")),
        ({"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
         normalizers.Replace(" ", "▁")),
        ({"type": "Replace", "pattern": {"String": "ab"}, "content": ""}, normalizers.Replace("ab", "")),
        ({"type": "Replace", "pattern": {"Regex": "[ \\t\\n]+"}, "content": " "},
         normalizers.Replace(Regex("[ \\t\\n]+"), " ")),
        ({"type": "Replace", "pattern": {"Regex": "[0-9]+"}, "content": "#$0"},
         normalizers.Replace(Regex("[0-9]+"), "#$0")),
        ({"type": "Replace", "pattern": {"Regex": "(?i)hello"}, "content": "bye"},
         normalizers.Replace(Regex("(?i)hello"), "bye")),
        ({"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"},
                                              {"type": "Strip", "strip_left": True, "strip_right": True}]},
         seq(normalizers.NFKC(), normalizers.Lowercase(), normalizers.Strip(True, True))),
        ({"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"Regex": " {2,}"},
                                               "content": " "}, {"type": "Prepend", "prepend": "_"}]},
         seq(normalizers.Replace(Regex(" {2,}"), " "), normalizers.Prepend("_"))),
    ]
    inputs = ["", " ", "Hello  World", "  padded  ", "ab ab", "ABC def", "ÉCOLE", "a\t\nb",
              "x1y22z333", "ﬁnd ①", "HELLO hello Hello", " nbsp ", "İstanbul",
              "ΣΣAΣ"]
    inputs += [fuzz_text(rng) for _ in range(150)]
    return {"inputs": inputs,
            "specs": [{"spec": spec, "outputs": [norm.normalize_str(t) for t in inputs]}
                      for spec, norm in specs]}


# --- the shipped checkpoints ------------------------------------------------------------------
ENCODE_POOL = (
    ["the", "refund", "payment", "account", "customer", "Please", "cancel", "my", "order", "I'm", "don't",
     "it's", "we've", "THEY'D", "12,345.67", "2026-10-11", "a@b.co", "https://example.com/x?y=1",
     "def f(x):", "{\"k\": [1, 2.5, null]}", "é", "café", "café", "naïve", "über",
     "中文测试", "日本語", "한국어", "مرحبا",
     "שלום", "नमस्ते", "สวัสดี",
     "Привет", "Γεια", "\U0001f600", "\U0001f468‍\U0001f469",
     "❤️", "\U0001f44d\U0001f3fd", "\U00010a00", "ೱ", "\U00016fe0", "\U0001e900",
     "▁", " ", "​", "‮", "﻿", "\u0000", "�", "½", "Ⅷ",
     "[MASK]", "[CLS]", "[SEP]", "[PAD]", "<mask>", "<bos>", "<eos>", "<pad>", "<unk>", "<s>", "</s>",
     "<0x41>", "<0xE2>", "ẞ", "İ", "ᾳ", "Å", "ﬁ"]
    + [" "] * 10 + ["  ", "\t", "\n", "\r\n", "   ", "　"]
)


def encode_text(rng):
    kind = rng.random()
    if kind < 0.08:
        return rng.choice("xqz中あ ") * rng.randrange(33, 220)
    if kind < 0.12:
        return "".join(rng.choice("abcdefghij") for _ in range(rng.randrange(40, 180)))
    return "".join(rng.choice(ENCODE_POOL) for _ in range(rng.randrange(0, 14)))


def checkpoint_tokenizer(name):
    from tokenizers import Tokenizer

    root = os.environ.get("LAYA_FIXTURE_CHECKPOINTS")
    if not root:
        raise SystemExit("gen_go_fixtures: set LAYA_FIXTURE_CHECKPOINTS to a directory holding the "
                         "english and multilingual checkpoints")
    path = os.path.join(root, name, "tokenizer", "tokenizer.json")
    if not os.path.isfile(path):
        raise SystemExit("gen_go_fixtures: no tokenizer at %s" % path)
    return Tokenizer.from_file(path)


def encode_family(rng):
    out = {}
    for name in ("english", "multilingual"):
        tok = checkpoint_tokenizer(name)
        texts = [encode_text(rng) for _ in range(700)]
        out[name] = {"texts": texts,
                     "ids": [tok.encode(t, add_special_tokens=False).ids for t in texts]}
    return out


def decode_family(rng):
    out = {}
    for name in ("english", "multilingual"):
        tok = checkpoint_tokenizer(name)
        size = tok.get_vocab_size(with_added_tokens=True)
        byte_ids = [tok.token_to_id("<0x%02X>" % b) for b in range(256)]
        byte_ids = [i for i in byte_ids if i is not None]
        low = list(range(0, 300))
        cases = []
        for _ in range(900):
            pool = rng.choice([byte_ids, low, None]) or list(range(size))
            ids = [rng.choice(pool) if pool is not None else rng.randrange(size)
                   for _ in range(rng.randrange(0, 10))]
            if rng.random() < 0.1:
                ids.append(size + rng.randrange(5))
            cases.append({"ids": ids,
                          "skipping": tok.decode(ids, skip_special_tokens=True),
                          "keeping": tok.decode(ids, skip_special_tokens=False)})
        out[name] = {"cases": cases}
    return out


GERMAN = "Ich kann mich nicht in mein Konto einloggen und das Passwort funktioniert nicht"


def state_family(rng):
    """How lang.py walks a state: Python's verdict on each container shape Go can spell."""
    from laya import lang

    self_list = [GERMAN]
    self_list.append(self_list)
    self_dict = {"a": GERMAN}
    self_dict["b"] = self_dict
    shapes = {
        "string": GERMAN,
        "map_of_slices": {"a": [GERMAN]},
        "slice_of_maps": [{"a": GERMAN}],
        "map_of_objects": {"a": {"b": GERMAN}},
        "slice_of_slices": [[GERMAN]],
        "tuple": (GERMAN,),
        "seven_deep": [[[[[[[GERMAN]]]]]]],
        "six_deep": [[[[[[GERMAN]]]]]],
        "numbers_and_none": [1, 2.5, None, True],
        "bytes": GERMAN.encode("utf-8"),
        "invalid_bytes": b"\xff\xfe",
        "mixed": {"k": [GERMAN, 3, {"x": None}]},
        "non_string_keys": {1: GERMAN},
        "self_ref_list": self_list,
        "self_ref_dict": self_dict,
    }
    out = {}
    for name, state in shapes.items():
        a = lang.analyse(state)
        out[name] = {"state_text": lang.state_text(state), "is_english": a["is_english"],
                     "language": a["language"], "script": a["script"]}
    return out


def invalid_utf8_family(rng):
    """CPython's bytes.decode('utf-8', 'replace') on random byte strings, and what each shipped
    tokenizer and lang.py make of the result."""
    from laya import lang

    pieces = [b"a", b"b", b" ", b"\xe2\x82\xac", b"\xe2\x82", b"\xe2", b"\xff", b"\xfe", b"\x80",
              b"\xc0\x80", b"\xed\xa0\x80", b"\xf0\x9f\x98\x80", b"\xf0\x9f\x98", b"\xf4\x90\x80\x80",
              b"\xc3\xa9", b"\xc3", b"\xe4\xb8\xad", b"Ich kann nicht"]
    cases = []
    for _ in range(3000):
        raw = b"".join(rng.choice(pieces) if rng.random() < 0.7 else bytes([rng.randrange(256)])
                       for _ in range(rng.randrange(0, 9)))
        cases.append(raw)
    decoded = [r.decode("utf-8", "replace") for r in cases]
    toks = {n: checkpoint_tokenizer(n) for n in ("english", "multilingual")}
    return {"hex": [r.hex() for r in cases], "decoded": decoded,
            "script": [lang.detect_script(d) for d in decoded],
            "state_text": [lang.state_text(d) for d in decoded],
            "ids": {n: [t.encode(d, add_special_tokens=False).ids for d in decoded[:600]]
                    for n, t in toks.items()}}


def build():
    import tokenizers

    rng = random.Random(SEED)
    return {
        "meta": {"seed": SEED, "tokenizers": tokenizers.__version__,
                 "cpython_unicode": unicodedata.unidata_version},
        "normalize": normalize_family(rng),
        "lowercase": lowercase_family(rng),
        "python_lower": sigma_family(rng),
        "single_word": single_word_family(rng),
        "pretokenize": pretokenize_family(rng),
        "normalizers": normalizers_family(rng),
        "encode_fuzz": encode_family(rng),
        "decode_fuzz": decode_family(rng),
        "state_shapes": state_family(rng),
        "invalid_utf8": invalid_utf8_family(rng),
    }


def render(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", metavar="DIR", default=DEFAULT_OUT,
                        help="write %s here (default laya-go/fixtures)" % NAME)
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the file in --out differs")
    args = parser.parse_args(argv)
    path = os.path.join(os.path.abspath(args.out), NAME)
    fresh = render(build())
    if args.check:
        try:
            with open(path, encoding="utf-8") as handle:
                current = handle.read()
        except OSError:
            current = None
        if current != fresh:
            print("gen_go_fixtures: %s is stale or missing; run laya-go/scripts/gen_go_fixtures.py"
                  % path, file=sys.stderr)
            return 1
        print("gen_go_fixtures: %s matches" % path)
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    print("wrote %s (%d bytes)" % (path, len(fresh.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
