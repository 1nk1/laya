package agent

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"reflect"
	"sync"
	"sync/atomic"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// ErrClosed is returned by an Agent that has been closed. A session closed underneath an open
// agent yields a wrapped infer.ErrClosed instead.
var ErrClosed = errors.New("agent: agent is closed")

// StateError identifies the state at fault in a PredictBatch call.
type StateError struct {
	// Index is the state's position in the states slice.
	Index int
	Err   error
}

// Error names the state and the cause.
func (e *StateError) Error() string { return fmt.Sprintf("agent: state %d: %v", e.Index, e.Err) }

// Unwrap returns the cause.
func (e *StateError) Unwrap() error { return e.Err }

// Agent answers typed questions about a state with a laya checkpoint, through an infer.Session.
// It is the port of laya's ONNXAgent, and it is safe for concurrent use. Use Open: the zero
// Agent is closed.
type Agent struct {
	tok   *tokenizer.Tokenizer
	padID int32
	cfg   Config
	temps *decode.Temperatures
	sess  infer.Session

	mu     sync.RWMutex // held for reading by every call, for writing by Close
	closed atomic.Bool  // set when Close starts, so new and queued calls return at once
}

// Option adjusts an Agent at Open. No options exist yet; the parameter is here so adding one is
// not a breaking change.
type Option func(*options)

type options struct{}

// Open loads rl_agent_config.json and the tokenizer from checkpoint, a directory holding a
// laya checkpoint, and takes ownership of s: on error it closes s, and otherwise Close does.
// The tokenizer must name the pad, CLS, SEP and mask tokens. The session must run the graph
// exported from that same checkpoint; nothing here can tell.
func Open(checkpoint fs.FS, s infer.Session, opts ...Option) (*Agent, error) {
	if s == nil {
		return nil, errors.New("agent: Open needs a session")
	}
	var o options
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	a, err := open(checkpoint, s)
	if err != nil {
		return nil, errors.Join(err, s.Close())
	}
	return a, nil
}

func open(checkpoint fs.FS, s infer.Session) (*Agent, error) {
	if checkpoint == nil {
		return nil, errors.New("agent: Open needs a checkpoint")
	}
	cfg, temps, err := loadConfig(checkpoint)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFS(checkpoint, ".")
	if err != nil {
		return nil, &ConfigError{Field: tokenizerField, Err: err}
	}
	var pad int32
	for _, t := range []struct {
		name string
		id   func() (int32, bool)
	}{{"pad", tok.Pad}, {"CLS", tok.CLS}, {"SEP", tok.SEP}, {"mask", tok.Mask}} {
		id, ok := t.id()
		if !ok {
			return nil, fmt.Errorf("agent: %w: no %s token", sequence.ErrMissingSpecialToken, t.name)
		}
		if t.name == "pad" {
			pad = id
		}
	}
	return &Agent{tok: tok, padID: pad, cfg: cfg, temps: temps, sess: s}, nil
}

// Config returns a copy of the checkpoint configuration the agent runs with.
func (a *Agent) Config() Config {
	cfg := a.cfg
	cfg.Warnings = append([]string(nil), cfg.Warnings...)
	return cfg
}

// Close marks the agent closed, so new and queued calls return ErrClosed at once, then waits for
// calls in flight and closes the session. Close blocks until calls in flight return; cancel
// their contexts to bound it. It is idempotent.
func (a *Agent) Close() error {
	first := a.closed.CompareAndSwap(false, true)
	a.mu.Lock()
	defer a.mu.Unlock()
	if !first || a.sess == nil {
		return nil
	}
	return a.sess.Close()
}

// CallOption adjusts one Predict or PredictBatch call.
type CallOption func(*call)

type call struct {
	lang  string
	batch int
	sort  bool
}

// WithLanguage names the request's language as a BCP 47 tag. Its primary subtag selects a
// language's temperatures when the agent has any; with none configured it changes nothing.
func WithLanguage(tag string) CallOption { return func(c *call) { c.lang = tag } }

// WithBatchSize caps the states sent per session run; 0 or less means all in one run. A run
// holds states x questions rows.
func WithBatchSize(states int) CallOption { return func(c *call) { c.batch = states } }

// WithSortByLength(true) pads less: states are ordered by length inside a lookahead of eight
// batches before they are chunked. It needs a batch size above 1 and below the number of states,
// and results stay in input order. A different batch shape can move a probability by a rounding
// step. WithSortByLength(false) is the default.
func WithSortByLength(on bool) CallOption { return func(c *call) { c.sort = on } }

// Predict answers every question about one state. Its errors are those of PredictBatch, except
// that a failing state's error is returned without the *StateError around it.
func (a *Agent) Predict(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (schema.Prediction, error) {
	preds, err := a.PredictBatch(ctx, []any{state}, qs, opts...)
	if err != nil {
		var se *StateError
		if errors.As(err, &se) {
			err = se.Err
		}
		return schema.Prediction{}, err
	}
	return preds[0], nil
}

// PredictBatch answers the questions for each state; results are in input order. A state is a
// string, or any value sequence.SerializeState accepts: a slice is a conversation whose end
// survives truncation. A nil state or nil pointer is refused; a nil slice or map is an empty
// list or dict. With no questions each state gets an empty answer and zero usage, with no model
// run.
//
// Errors may be wrapped; match them with errors.Is and errors.As:
//   - ErrClosed, once Close has been called;
//   - ctx.Err(), checked on entry, before each state is encoded and between runs;
//   - *StateError, naming the state, around sequence.ErrUnsupportedState (also for a nil
//     state), sequence.ErrMissingSpecialToken or a *sequence.MarkersError with its ID set;
//   - *RunError, around whatever the session returned (so errors.Is finds context.Canceled and
//     infer.ErrClosed through it), a *infer.ShapeError, or a logit that cannot be decoded.
func (a *Agent) PredictBatch(ctx context.Context, states []any, qs schema.Questions, opts ...CallOption) ([]schema.Prediction, error) {
	var c call
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if a.sess == nil || a.closed.Load() {
		return nil, ErrClosed
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.closed.Load() {
		return nil, ErrClosed
	}
	for i, state := range states {
		if isNil(state) {
			return nil, &StateError{Index: i, Err: fmt.Errorf("is nil; pass a string, a map or a slice: %w", sequence.ErrUnsupportedState)}
		}
	}
	if len(states) == 0 {
		return []schema.Prediction{}, nil
	}
	if qs.Len() == 0 {
		preds := make([]schema.Prediction, len(states))
		for i := range preds {
			preds[i] = schema.Prediction{Model: schema.Model, Answers: schema.Answers{},
				Usage: schema.Usage{TruncatedQuestions: []string{}}}
		}
		return preds, nil
	}
	return a.predictChunks(ctx, states, qs, c)
}

// isNil reports an untyped nil or a nil pointer; a nil slice or map is an empty list or dict.
func isNil(state any) bool {
	if state == nil {
		return true
	}
	rv := reflect.ValueOf(state)
	return rv.Kind() == reflect.Pointer && rv.IsNil()
}
