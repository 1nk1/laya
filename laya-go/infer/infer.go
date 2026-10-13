package infer

import (
	"context"
	"errors"
	"fmt"
)

// ErrClosed is returned by a Session that has been closed.
var ErrClosed = errors.New("laya: session is closed")

// Batch is the graph's five inputs, each flattened row-major, as onnxruntime tensors take them.
type Batch struct {
	InputIDs              []int64 // [Rows*Length]; padding is the pad id
	AttentionMask         []int64 // [Rows*Length]; 1 for real tokens
	MarkerPos             []int64 // [Rows*Markers]
	MarkerMask            []bool  // [Rows*Markers]; true for real markers
	QType                 []int64 // [Rows]; the question type code
	Rows, Length, Markers int
}

// Output is the graph's two outputs, row-major: Logits is [Rows*Markers], ActLogits [Rows*2].
type Output struct {
	Logits, ActLogits []float32
}

// Session runs one collated batch. Implementations must be safe for concurrent Run.
type Session interface {
	// Run returns the logits for b. It should return ctx.Err() if ctx ends before the run does.
	Run(ctx context.Context, b Batch) (Output, error)
	// Close releases the runtime. It is idempotent; Run after Close returns ErrClosed.
	Close() error
}

// ShapeError is returned when a slice's length does not match the batch it belongs to.
type ShapeError struct {
	// Name is the tensor: "input_ids", "logits" and so on.
	Name string
	// Got is the slice's length, Want the length the batch's dimensions call for.
	Got, Want int
}

// Error names the tensor and both lengths.
func (e *ShapeError) Error() string {
	return fmt.Sprintf("laya: %s has %d values, want %d", e.Name, e.Got, e.Want)
}

// Validate reports a *ShapeError when a slice's length disagrees with Rows, Length and Markers,
// and a plain error when a dimension is negative or Markers is below 2.
func (b Batch) Validate() error {
	if b.Rows < 0 || b.Length < 0 || b.Markers < 0 {
		return fmt.Errorf("laya: batch has negative dimensions %d x %d x %d", b.Rows, b.Length, b.Markers)
	}
	if b.Rows > 0 && b.Markers < 2 {
		return fmt.Errorf("laya: batch has %d marker slots, the graph needs at least 2", b.Markers)
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"input_ids", len(b.InputIDs), b.Rows * b.Length},
		{"attention_mask", len(b.AttentionMask), b.Rows * b.Length},
		{"marker_pos", len(b.MarkerPos), b.Rows * b.Markers},
		{"marker_mask", len(b.MarkerMask), b.Rows * b.Markers},
		{"qtype", len(b.QType), b.Rows},
	} {
		if c.got != c.want {
			return &ShapeError{Name: c.name, Got: c.got, Want: c.want}
		}
	}
	return nil
}

// Check reports a *ShapeError when o does not hold Rows*Markers logits and Rows*2 action logits
// for b.
func (o Output) Check(b Batch) error {
	if len(o.Logits) != b.Rows*b.Markers {
		return &ShapeError{Name: "logits", Got: len(o.Logits), Want: b.Rows * b.Markers}
	}
	if len(o.ActLogits) != b.Rows*2 {
		return &ShapeError{Name: "act_logits", Got: len(o.ActLogits), Want: b.Rows * 2}
	}
	return nil
}
