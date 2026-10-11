package lang

import (
	"encoding/json"
	"strings"
	"testing"
)

// JSON decoded into an Object keeps the key order Python's dict would have, and that order
// decides which text a long state is read from first.
func TestObjectUnmarshalJSONKeepsOrder(t *testing.T) {
	var o Object
	doc := `{"z":"first","a":{"y":"second","b":["third",{"q":"fourth"}]},"m":"fifth","n":3,"o":null}`
	if err := json.Unmarshal([]byte(doc), &o); err != nil {
		t.Fatal(err)
	}
	if got := StateText(o); got != "first second third fourth fifth" {
		t.Errorf("StateText = %q", got)
	}
	keys := make([]string, len(o))
	for i, m := range o {
		keys[i] = m.Key
	}
	if strings.Join(keys, ",") != "z,a,m,n,o" {
		t.Errorf("keys = %v", keys)
	}
	if _, ok := o[1].Value.(Object); !ok {
		t.Errorf("a nested object must be an Object, got %T", o[1].Value)
	}
	// The same document through a map is read in sorted key order, which is the difference.
	var m map[string]any
	if err := json.Unmarshal([]byte(doc), &m); err != nil {
		t.Fatal(err)
	}
	if StateText(m) == StateText(o) {
		t.Error("a map and an Object of the same JSON should read in different orders here")
	}
}

func TestObjectUnmarshalJSONRefusesWhatIsNotAnObject(t *testing.T) {
	var o Object
	for _, doc := range []string{`[1]`, `"x"`, `{"a":`, `{"a":1`, `{`} {
		if err := json.Unmarshal([]byte(doc), &o); err == nil {
			t.Errorf("%s must be refused", doc)
		}
	}
	var holder struct{ State Object }
	if err := json.Unmarshal([]byte(`{"State":{"b":"1","a":"2"}}`), &holder); err != nil || StateText(holder.State) != "1 2" {
		t.Errorf("Object inside a struct: %v %q", err, StateText(holder.State))
	}
}

func TestScriptConstantsAreTheNamesPythonUses(t *testing.T) {
	for script, text := range map[Script]string{
		ScriptLatin: "hello", ScriptHan: "你好吗", ScriptDevanagari: "नमस्ते", ScriptCyrillic: "привет",
		ScriptGreek: "γεια", ScriptHangul: "안녕하세요", ScriptKana: "こんにちは", ScriptUnknown: "123",
	} {
		if got := DetectScript(text); got != script {
			t.Errorf("DetectScript(%q) = %q, want %q", text, got, script)
		}
	}
	if string(ScriptOther) != "other" || len(scriptNames) != 25 {
		t.Errorf("ScriptOther = %q, %d named scripts", ScriptOther, len(scriptNames))
	}
}
