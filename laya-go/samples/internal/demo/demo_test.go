package demo

import (
	"reflect"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// The expected text is json.dumps(state, ensure_ascii=False) from Python.
func TestStateSerializesInPythonDictOrder(t *testing.T) {
	got, err := sequence.SerializeState(State("Charged twice for order 8812", "I was billed two times this morning."))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"subject": "Charged twice for order 8812", "body": "I was billed two times this morning."}`
	if got != want {
		t.Fatalf("state = %s, want %s", got, want)
	}
}

func TestQuestionsAreTheEmailPreset(t *testing.T) {
	qs, err := Questions()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"category", "is_spam", "urgency", "needs_reply"}
	if got := qs.IDs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("ids = %v, want %v", got, want)
	}
}
