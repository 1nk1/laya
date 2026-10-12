// Package pyvalue reads, writes and describes the values laya's Python code handles as dicts,
// lists, strs, ints and floats: the state of a request, a question's criteria, a definition in
// the wire form.
//
// It sits above pyjson (which writes the text) and adds what a Go caller needs to reach it: a
// decoder that keeps object order and number spelling as json.loads does, a normalizer that maps
// the Go shapes a caller can hold (maps, slices, pointers, json.Number, lang.Object) onto what
// pyjson accepts, and the repr and str of a decoded value, which laya puts in its error messages.
package pyvalue

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/lang"
)

// MaxDepth bounds the nesting of a decoded or normalized value, as Python's recursion limit
// does, so a hostile document or a self-referential pointer ends in an error.
const MaxDepth = 512

// ErrUnsupported is wrapped when a Go value has no Python equivalent json.dumps would write.
var ErrUnsupported = errors.New("value has no JSON equivalent in Python")

// Parse decodes one JSON document as json.loads does: an object is a pyjson.Object with string
// keys in document order (a repeated key keeps its first position and takes its last value),
// numbers are json.Number so that 2 stays an int and 2.0 a float, and anything after the value
// is an error. NaN, Infinity and invalid UTF-8 are refused.
func Parse(data []byte) (any, error) {
	dec, err := newDecoder(data)
	if err != nil {
		return nil, err
	}
	v, err := decodeValue(dec, 0, false)
	if err != nil {
		return nil, err
	}
	return v, endOfDocument(dec)
}

// ParseMembers decodes a document whose top level is an object and returns its members as
// written, repeated keys included, so a caller that refuses a repeat can see it. Nested objects
// are decoded as Parse does.
func ParseMembers(data []byte) ([]pyjson.Member, error) {
	dec, err := newDecoder(data)
	if err != nil {
		return nil, err
	}
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if tok != json.Delim('{') {
		return nil, fmt.Errorf("want a JSON object, got %s", describeToken(tok))
	}
	members, err := decodeMembers(dec, 1, true)
	if err != nil {
		return nil, err
	}
	return members, endOfDocument(dec)
}

func newDecoder(data []byte) (*json.Decoder, error) {
	if !utf8.Valid(data) {
		return nil, errors.New("JSON text is not valid UTF-8")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	return dec, nil
}

func endOfDocument(dec *json.Decoder) error {
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("data after the JSON value")
	}
	return nil
}

func describeToken(tok json.Token) string {
	if d, ok := tok.(json.Delim); ok {
		return string(d)
	}
	return fmt.Sprintf("%v", tok)
}

func decodeValue(dec *json.Decoder, depth int, keepRepeats bool) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("JSON nested deeper than %d levels", MaxDepth)
	}
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	if delim == '{' {
		members, err := decodeMembers(dec, depth+1, keepRepeats)
		if err != nil {
			return nil, err
		}
		return pyjson.Object(members), nil
	}
	list := []any{}
	for dec.More() {
		v, err := decodeValue(dec, depth+1, keepRepeats)
		if err != nil {
			return nil, err
		}
		list = append(list, v)
	}
	_, err = dec.Token()
	return list, err
}

// decodeMembers reads members up to the closing brace of an object whose opening brace was read.
func decodeMembers(dec *json.Decoder, depth int, keepRepeats bool) ([]pyjson.Member, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("JSON nested deeper than %d levels", MaxDepth)
	}
	members := []pyjson.Member{}
	at := map[string]int{}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key := keyTok.(string)
		value, err := decodeValue(dec, depth, false)
		if err != nil {
			return nil, err
		}
		if i, seen := at[key]; seen && !keepRepeats {
			members[i].Value = value
			continue
		}
		at[key] = len(members)
		members = append(members, pyjson.Member{Key: key, Value: value})
	}
	_, err := dec.Token()
	return members, err
}

// Number converts a json.Number to what json.loads would have produced: an int (int64, or
// *big.Int when it does not fit) when the text has no fraction or exponent, else a float64.
func Number(n json.Number) (any, error) {
	text := string(n)
	if !strings.ContainsAny(text, ".eE") {
		if i, err := strconv.ParseInt(text, 10, 64); err == nil {
			return i, nil
		}
		b, ok := new(big.Int).SetString(text, 10)
		if !ok {
			return nil, fmt.Errorf("%q is not a JSON number", text)
		}
		return b, nil
	}
	f, err := strconv.ParseFloat(text, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return nil, fmt.Errorf("%q is not a JSON number", text)
	}
	return f, nil // out of range is +-Inf, as float("1e400") is in Python
}

// Normalize maps v onto the shapes pyjson writes: nil, bool, string, integers, floats, *big.Int,
// []any, pyjson.Object and map[string]any. Pointers are followed, a json.Number becomes an int or
// a float by its spelling, lang.Object keeps its order, a map with string keys is written in
// sorted key order, and a slice or array (but not bytes) is a list. A nil slice is an empty list
// and a nil map an empty object. Anything else, a struct or a channel or a map with other keys,
// is refused rather than stringified.
func Normalize(v any) (any, error) { return normalize(v, 0) }

func normalize(v any, depth int) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("%w: nested deeper than %d levels", ErrUnsupported, MaxDepth)
	}
	switch x := v.(type) {
	case nil, bool, string, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64,
		*big.Int, float32, float64:
		return v, nil
	case json.Number:
		return Number(x)
	case []string:
		return x, nil
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			n, err := normalize(item, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case pyjson.Object:
		out := make(pyjson.Object, len(x))
		for i, m := range x {
			n, err := normalize(m.Value, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: m.Key, Value: n}
		}
		return out, nil
	case lang.Object:
		out := make(pyjson.Object, len(x))
		for i, m := range x {
			n, err := normalize(m.Value, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: m.Key, Value: n}
		}
		return out, nil
	}
	return normalizeReflect(reflect.ValueOf(v), depth)
}

func normalizeReflect(rv reflect.Value, depth int) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("%w: nested deeper than %d levels", ErrUnsupported, MaxDepth)
	}
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return nil, nil
		}
		return normalize(rv.Elem().Interface(), depth+1)
	case reflect.String:
		return rv.String(), nil
	case reflect.Bool:
		return rv.Bool(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint(), nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	case reflect.Slice, reflect.Array:
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			return nil, fmt.Errorf("%w: %s is bytes, which json.dumps refuses", ErrUnsupported, rv.Type())
		}
		out := make([]any, rv.Len())
		for i := range out {
			n, err := normalize(rv.Index(i).Interface(), depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf("%w: %s has non-string keys, whose order Go does not define",
				ErrUnsupported, rv.Type())
		}
		keys := rv.MapKeys()
		sort.Slice(keys, func(i, j int) bool { return keys[i].String() < keys[j].String() })
		out := make(pyjson.Object, len(keys))
		for i, k := range keys {
			n, err := normalize(rv.MapIndex(k).Interface(), depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: k.String(), Value: n}
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: %s", ErrUnsupported, rv.Type())
}

// Dumps is json.dumps(v, ensure_ascii=False) for a Go value, with the default ", " and ": "
// separators. It is Normalize followed by pyjson.Dumps.
func Dumps(v any) (string, error) {
	n, err := Normalize(v)
	if err != nil {
		return "", err
	}
	s, err := pyjson.Dumps(n)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrUnsupported, err)
	}
	return s, nil
}

// TypeName is type(v).__name__ for a decoded JSON value.
func TypeName(v any) string {
	switch x := v.(type) {
	case nil:
		return "NoneType"
	case bool:
		return "bool"
	case string:
		return "str"
	case json.Number:
		if strings.ContainsAny(string(x), ".eE") {
			return "float"
		}
		return "int"
	case []any:
		return "list"
	case pyjson.Object:
		return "dict"
	}
	return fmt.Sprintf("%T", v)
}

// Repr is repr(v) for a decoded JSON value: None, True, 'text', 1, 2.5, [..] and {'k': v}.
func Repr(v any) string {
	var b strings.Builder
	writeRepr(&b, v)
	return b.String()
}

func writeRepr(b *strings.Builder, v any) {
	switch x := v.(type) {
	case nil:
		b.WriteString("None")
	case bool:
		if x {
			b.WriteString("True")
		} else {
			b.WriteString("False")
		}
	case string:
		b.WriteString(pyjson.Repr(x))
	case json.Number:
		b.WriteString(Str(x))
	case []any:
		b.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeRepr(b, item)
		}
		b.WriteByte(']')
	case pyjson.Object:
		b.WriteByte('{')
		for i, m := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeRepr(b, m.Key)
			b.WriteString(": ")
			writeRepr(b, m.Value)
		}
		b.WriteByte('}')
	default:
		fmt.Fprint(b, v)
	}
}

// Str is str(v) for a decoded JSON scalar: the text itself for a string, True and False, the
// digits of an int, and the repr of a float (1.0, 1e+16, inf).
func Str(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case json.Number:
		n, err := Number(x)
		if err != nil {
			return string(x)
		}
		switch n := n.(type) {
		case int64:
			return strconv.FormatInt(n, 10)
		case *big.Int:
			return n.String()
		case float64:
			return floatStr(n)
		}
	case float64:
		return floatStr(x)
	}
	return Repr(v)
}

func floatStr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	return pyjson.ReprFloat(f)
}

// KeyOf is a string that is equal for two decoded scalars exactly when Python's == and hash treat
// them as one dict key: 1, 1.0 and True are one key, and a str never equals a number.
func KeyOf(v any) string {
	switch x := v.(type) {
	case string:
		return "s:" + x
	case bool:
		if x {
			return "n:1"
		}
		return "n:0"
	case json.Number:
		n, err := Number(x)
		if err != nil {
			return "x:" + string(x)
		}
		switch n := n.(type) {
		case int64:
			return "n:" + strconv.FormatInt(n, 10)
		case *big.Int:
			return "n:" + n.String()
		case float64:
			if math.IsInf(n, 0) {
				return "n:" + floatStr(n)
			}
			r := new(big.Rat).SetFloat64(n) // the exact binary value, not its shortest spelling
			if r.IsInt() {
				return "n:" + r.Num().String()
			}
			return "n:" + r.String()
		}
	}
	return "x:" + Repr(v)
}
