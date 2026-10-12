package fixtures

import (
	"encoding/json"
	"strconv"
	"strings"
)

// Toy special-token ids of the tokenizer ToyTokenizer returns.
const (
	ToyPad  = 256
	ToyCLS  = 257
	ToySEP  = 258
	ToyMask = 259
)

// ToyTokenizer returns the tokenizer.json and tokenizer_config.json of a tiny byte-level BPE
// tokenizer: one id per byte (0-255), [PAD] [CLS] [SEP] [MASK] at 256-259, and the merges
// "t h" (260) and "i n" (261). Examples and unit tests use it where no checkpoint is at hand.
func ToyTokenizer() (tokenizerJSON, configJSON []byte) {
	// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest move above U+0100.
	var printable []int
	for b := '!'; b <= '~'; b++ {
		printable = append(printable, int(b))
	}
	for b := 0xA1; b <= 0xFF; b++ {
		if b != 0xAD {
			printable = append(printable, b)
		}
	}
	runes := make(map[int]rune, 256)
	for _, b := range printable {
		runes[b] = rune(b)
	}
	next := rune(256)
	for b := 0; b < 256; b++ {
		if _, ok := runes[b]; !ok {
			runes[b] = next
			next++
		}
	}
	vocab := make([]string, 0, 262)
	for b := 0; b < 256; b++ {
		key, _ := json.Marshal(string(runes[b]))
		vocab = append(vocab, string(key)+":"+strconv.Itoa(b))
	}
	vocab = append(vocab, `"[PAD]":256`, `"[CLS]":257`, `"[SEP]":258`, `"[MASK]":259`, `"th":260`, `"in":261`)
	var added []string
	for i, name := range []string{"[PAD]", "[CLS]", "[SEP]", "[MASK]"} {
		added = append(added, `{"id":`+strconv.Itoa(256+i)+`,"content":"`+name+
			`","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}`)
	}
	spec := `{"added_tokens":[` + strings.Join(added, ",") + `],` +
		`"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},` +
		`"decoder":{"type":"ByteLevel"},` +
		`"model":{"type":"BPE","vocab":{` + strings.Join(vocab, ",") + `},"merges":["t h","i n"]}}`
	config := `{"cls_token":"[CLS]","sep_token":"[SEP]","mask_token":"[MASK]","pad_token":"[PAD]"}`
	return []byte(spec), []byte(config)
}
