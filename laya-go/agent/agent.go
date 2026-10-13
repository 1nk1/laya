package agent

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"sync"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// ErrClosed is returned by an Agent that has been closed.
var ErrClosed = errors.New("laya: agent is closed")

// Agent answers typed questions about a state with a laya checkpoint, through an infer.Session.
// It is the port of laya's ONNXAgent, and it is safe for concurrent use.
type Agent struct {
	tok   *tokenizer.Tokenizer
	padID int32
	cfg   Config
	temps *decode.Temperatures
	sess  infer.Session

	mu     sync.RWMutex // held for reading by every call, for writing by Close
	closed bool
}

// Open loads rl_agent_config.json and the tokenizer from checkpoint, a directory holding a
// laya checkpoint, and takes ownership of s: on error it closes s, and otherwise Close does.
// The session must run the graph exported from that same checkpoint; nothing here can tell.
func Open(checkpoint fs.FS, s infer.Session) (*Agent, error) {
	if s == nil {
		return nil, errors.New("laya: Open needs a session")
	}
	a, err := open(checkpoint, s)
	if err != nil {
		_ = s.Close()
		return nil, err
	}
	return a, nil
}

func open(checkpoint fs.FS, s infer.Session) (*Agent, error) {
	cfg, temps, err := loadConfig(checkpoint)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFS(checkpoint, ".")
	if err != nil {
		return nil, fmt.Errorf("laya: tokenizer: %w", err)
	}
	pad, ok := tok.Pad()
	if !ok {
		return nil, errors.New("laya: tokenizer names no pad token")
	}
	return &Agent{tok: tok, padID: pad, cfg: cfg, temps: temps, sess: s}, nil
}

// Config is the checkpoint configuration in use, after clamping.
func (a *Agent) Config() Config {
	cfg := a.cfg
	cfg.Warnings = append([]string(nil), cfg.Warnings...)
	return cfg
}

// Close waits for calls in flight, then closes the session. It is idempotent; a later Predict
// returns ErrClosed.
func (a *Agent) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
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

// WithSortByLength pads less: states are ordered by length inside a lookahead of eight batches
// before they are chunked. It needs a batch size above 1 and below the number of states, and
// results stay in input order. A different batch shape can move a probability by a rounding step.
func WithSortByLength() CallOption { return func(c *call) { c.sort = true } }

// Predict answers every question about one state. Its errors are those of PredictBatch.
func (a *Agent) Predict(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (schema.Prediction, error) {
	preds, err := a.PredictBatch(ctx, []any{state}, qs, opts...)
	if err != nil {
		return schema.Prediction{}, err
	}
	return preds[0], nil
}

// PredictBatch answers the questions for each state; results are in input order. A state is a
// string, or any value sequence.SerializeState accepts: a slice is a conversation whose end
// survives truncation. Errors pass through unchanged: *schema.QuestionError,
// *sequence.MarkersError (ID set), sequence.ErrUnsupportedState and ctx.Err(). With no questions
// each state gets an empty answer and zero usage, with no model run.
func (a *Agent) PredictBatch(ctx context.Context, states []any, qs schema.Questions, opts ...CallOption) ([]schema.Prediction, error) {
	var c call
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.closed {
		return nil, ErrClosed
	}
	for i, state := range states {
		if state == nil {
			return nil, fmt.Errorf("laya: state %d is nil; pass a string, a map or a slice", i)
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
	return a.infer(ctx, states, qs, c)
}
