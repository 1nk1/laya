package schema

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
)

// Model is the model name laya's ONNX backend reports, so a consumer cannot tell the ports apart.
const Model = "laya-rl-agent-onnx"

// Answered is one question's answer, with the question id.
type Answered struct {
	ID     string
	Answer Answer
}

// Answers are one state's answers in the order the questions were asked. They marshal as one
// JSON object in that order.
type Answers []Answered

// Get is the answer to the question with id.
func (as Answers) Get(id string) (Answer, bool) {
	for _, a := range as {
		if a.ID == id {
			return a.Answer, true
		}
	}
	return nil, false
}

// MarshalJSON writes {"id": answer, ...} in order.
func (as Answers) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('{')
	for i, a := range as {
		if i > 0 {
			b.WriteByte(',')
		}
		if a.Answer == nil {
			return nil, fmt.Errorf("schema: answer %s is nil", pyjson.Repr(a.ID))
		}
		id, err := pyjson.Dumps(a.ID)
		if err != nil {
			return nil, err
		}
		m, ok := a.Answer.(json.Marshaler)
		if !ok {
			return nil, fmt.Errorf("schema: answer %s is a %T, which cannot be marshaled", pyjson.Repr(a.ID), a.Answer)
		}
		body, err := m.MarshalJSON()
		if err != nil {
			return nil, err
		}
		b.WriteString(id + ":")
		b.Write(body)
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// UnmarshalJSON reads an answers object in document order.
func (as *Answers) UnmarshalJSON(data []byte) error {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return fmt.Errorf("schema: answers: %w", err)
	}
	obj, ok := v.(pyjson.Object)
	if !ok {
		return fmt.Errorf("schema: answers are a JSON object, got %s", pyvalue.TypeName(v))
	}
	out := make(Answers, len(obj))
	for i, m := range obj {
		a, err := answerFrom(m.Value)
		if err != nil {
			return fmt.Errorf("answer %s: %w", pyjson.Repr(m.Key.(string)), err)
		}
		out[i] = Answered{ID: m.Key.(string), Answer: a}
	}
	*as = out
	return nil
}

// Prediction is one state's answers, with what they cost: laya's predict() result.
type Prediction struct {
	// Model is the runtime that produced it; Model for the ONNX backend.
	Model   string  `json:"model"`
	Answers Answers `json:"answers"`
	Usage   Usage   `json:"usage"`
}

// Usage is what one prediction cost, and what it silently lost.
//
// The second half is the point. A state longer than the room the question's head leaves is
// truncated without an error, and a caller cannot reconstruct that from outside: the budget is in
// tokens, and the room moves per checkpoint and per question. Truncated and TruncatedQuestions
// report it.
type Usage struct {
	// InputTokens are the non-padding tokens fed to the encoder across every row of this state.
	// A shared state is encoded once per question, so it is counted once per question.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is always 0: laya is not a generator.
	OutputTokens int `json:"output_tokens"`
	// StateTokens is how many tokens the state serialized to, before any budget.
	StateTokens int `json:"state_tokens"`
	// StateTokensDropped is the worst case across the questions, which share a state but not a
	// head budget.
	StateTokensDropped int `json:"state_tokens_dropped"`
	// Truncated reports whether any question dropped state tokens.
	Truncated bool `json:"truncated"`
	// TruncatedQuestions are the ids of the questions that did, in request order.
	TruncatedQuestions []string `json:"truncated_questions"`
	// CollapsedOptions are the questions whose options no longer have a token span each, keyed
	// by question id. nil unless the head budget collapsed some; laya omits the key then.
	CollapsedOptions CollapsedQuestions `json:"options,omitempty"`
}

// MarshalJSON writes laya's usage dict. truncated_questions is always a list, as Python's is,
// never null.
func (u Usage) MarshalJSON() ([]byte, error) {
	type plain Usage // no methods, so no recursion
	if u.TruncatedQuestions == nil {
		u.TruncatedQuestions = []string{}
	}
	return json.Marshal(plain(u))
}

// CollapsedQuestion is one question's CollapsedOptions report, with the question id.
type CollapsedQuestion struct {
	ID string
	CollapsedOptions
}

// CollapsedQuestions are the reports of the questions whose options collapsed, in request order
// (laya's dict keeps it, and so does the JSON written here). The zero value is none.
type CollapsedQuestions []CollapsedQuestion

// Get is the report for the question with id.
func (cs CollapsedQuestions) Get(id string) (CollapsedOptions, bool) {
	for _, c := range cs {
		if c.ID == id {
			return c.CollapsedOptions, true
		}
	}
	return CollapsedOptions{}, false
}

// MarshalJSON writes {"id": {"total": n, "distinct": n, "tokens_per_option": n|null}, ...} in
// order.
func (cs CollapsedQuestions) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('{')
	for i, c := range cs {
		if i > 0 {
			b.WriteByte(',')
		}
		id, err := pyjson.Dumps(c.ID)
		if err != nil {
			return nil, err
		}
		body, err := json.Marshal(c.CollapsedOptions)
		if err != nil {
			return nil, err
		}
		b.WriteString(id + ":")
		b.Write(body)
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// UnmarshalJSON reads the object MarshalJSON writes, keeping its order.
func (cs *CollapsedQuestions) UnmarshalJSON(data []byte) error {
	members, err := pyvalue.ParseMembers(data)
	if err != nil {
		return fmt.Errorf("schema: options: %w", err)
	}
	out := make(CollapsedQuestions, len(members))
	for i, m := range members {
		body, err := json.Marshal(toPlain(m.Value))
		if err != nil {
			return err
		}
		out[i].ID = m.Key.(string)
		if err := json.Unmarshal(body, &out[i].CollapsedOptions); err != nil {
			return fmt.Errorf("options %s: %w", pyjson.Repr(out[i].ID), err)
		}
	}
	*cs = out
	return nil
}

// toPlain turns a decoded value into one encoding/json can write.
func toPlain(v any) any {
	switch x := v.(type) {
	case pyjson.Object:
		m := make(map[string]any, len(x))
		for _, member := range x {
			m[member.Key.(string)] = toPlain(member.Value)
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = toPlain(item)
		}
		return out
	}
	return v
}

// CollapsedOptions reports a question whose options the head budget made indistinguishable.
type CollapsedOptions struct {
	// Total is how many options the question defines, not how many markers reached the input.
	Total int `json:"total"`
	// Distinct is how many still have a token span of their own.
	Distinct int `json:"distinct"`
	// TokensPerOption is the cap the head budget applied, or nil when the collapse came from the
	// fixed 48-token option cap alone.
	TokensPerOption *int `json:"tokens_per_option"`
}
