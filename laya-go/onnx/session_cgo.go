//go:build cgo

package onnx

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	ort "github.com/yalue/onnxruntime_go"
)

// Session runs a fused laya graph on ONNX Runtime's CPU provider. It is safe for concurrent use.
type Session struct {
	mu      sync.RWMutex // read-held by Run, write-held by Close
	closed  bool
	s       *ort.DynamicAdvancedSession
	version string
	runs    atomic.Int64 // model runs started, for tests
}

var _ infer.Session = (*Session)(nil)

var (
	inputNames  = []string{infer.NameInputIDs, infer.NameAttentionMask, infer.NameMarkerPos, infer.NameMarkerMask, infer.NameQType}
	outputNames = []string{infer.NameLogits, infer.NameActLogits}
)

// Open loads a fused laya.onnx, whose .data sidecar must sit beside it, on the CPU provider.
// A graph that is missing, split, or declares other tensors is a *GraphError.
func Open(graphPath string, opts ...Option) (*Session, error) {
	o, err := resolve(opts)
	if err != nil {
		return nil, err
	}
	if err := checkPath(graphPath); err != nil {
		return nil, err
	}
	version, err := acquire(o.library)
	if err != nil {
		return nil, err
	}
	s, err := load(graphPath, o)
	if err != nil {
		return nil, errors.Join(err, release())
	}
	s.version = version
	return s, nil
}

func load(path string, o options) (*Session, error) {
	in, out, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: cannot read %s: %w", path, err)
	}
	if err := checkGraph(infos(in), infos(out)); err != nil {
		var ge *GraphError
		if errors.As(err, &ge) {
			ge.Path = path
		}
		return nil, err
	}
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnx: session options: %w", err)
	}
	defer so.Destroy()
	if err := so.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, fmt.Errorf("onnx: optimization level: %w", err)
	}
	if o.intra > 0 {
		if err := so.SetIntraOpNumThreads(o.intra); err != nil {
			return nil, fmt.Errorf("onnx: intra-op threads: %w", err)
		}
	}
	if o.inter > 0 {
		if err := so.SetInterOpNumThreads(o.inter); err != nil {
			return nil, fmt.Errorf("onnx: inter-op threads: %w", err)
		}
	}
	ds, err := ort.NewDynamicAdvancedSession(path, inputNames, outputNames, so)
	if err != nil {
		return nil, fmt.Errorf("onnx: cannot load %s (its .data sidecar must sit beside it): %w", path, err)
	}
	return &Session{s: ds}, nil
}

func infos(in []ort.InputOutputInfo) []tensorInfo {
	out := make([]tensorInfo, len(in))
	for i, t := range in {
		out[i] = tensorInfo{Name: t.Name, Dims: []int64(t.Dimensions), DType: "non-tensor"}
		if t.OrtValueType != ort.ONNXTypeTensor {
			continue
		}
		switch t.DataType {
		case ort.TensorElementDataTypeInt64:
			out[i].DType = "int64"
		case ort.TensorElementDataTypeBool:
			out[i].DType = "bool"
		case ort.TensorElementDataTypeFloat:
			out[i].DType = "float32"
		default:
			out[i].DType = t.DataType.String()
		}
	}
	return out
}

// RuntimeVersion is the version of the ONNX Runtime library in use, such as "1.30.0".
func (s *Session) RuntimeVersion() string { return s.version }

// Close waits for runs in flight, releases the graph and drops its hold on ONNX Runtime, which
// is unloaded when the last session closes. It is idempotent.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	err := s.s.Destroy()
	s.s = nil
	return errors.Join(err, release())
}

// Run runs one batch. A context that ends mid-run terminates the run and Run returns an error
// wrapping ctx.Err(). Inputs are not modified or retained, and the returned slices are the
// caller's.
func (s *Session) Run(ctx context.Context, b infer.Batch) (infer.Output, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return infer.Output{}, infer.ErrClosed
	}
	if err := b.Validate(); err != nil {
		return infer.Output{}, err
	}
	if err := ctx.Err(); err != nil {
		return infer.Output{}, fmt.Errorf("onnx: run not started: %w", err)
	}
	if b.Rows == 0 {
		return infer.Output{}, nil
	}

	var vals []ort.Value
	defer func() {
		for _, v := range vals {
			if v != nil {
				v.Destroy()
			}
		}
	}()
	rows, length, markers := int64(b.Rows), int64(b.Length), int64(b.Markers)
	ids, err := ort.NewTensor(ort.NewShape(rows, length), b.InputIDs)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameInputIDs, err)
	}
	vals = append(vals, ids)
	mask, err := ort.NewTensor(ort.NewShape(rows, length), b.AttentionMask)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameAttentionMask, err)
	}
	vals = append(vals, mask)
	pos, err := ort.NewTensor(ort.NewShape(rows, markers), b.MarkerPos)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameMarkerPos, err)
	}
	vals = append(vals, pos)
	mm, err := ort.NewTensor(ort.NewShape(rows, markers), b.MarkerMask)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameMarkerMask, err)
	}
	vals = append(vals, mm)
	qt, err := ort.NewTensor(ort.NewShape(rows), b.QType)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameQType, err)
	}
	vals = append(vals, qt)
	inputs := append([]ort.Value(nil), vals...)

	outputs := []ort.Value{nil, nil}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	s.runs.Add(1)
	if err := s.execute(ctx, inputs, outputs); err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return infer.Output{}, fmt.Errorf("onnx: run cancelled: %w", cerr)
		}
		return infer.Output{}, fmt.Errorf("onnx: run failed on %d rows of %d tokens: %w", b.Rows, b.Length, err)
	}
	logits, lok := outputs[0].(*ort.Tensor[float32])
	act, aok := outputs[1].(*ort.Tensor[float32])
	if !lok || !aok {
		return infer.Output{}, fmt.Errorf("onnx: outputs are %T and %T, want float32 tensors", outputs[0], outputs[1])
	}
	return collect(b, outTensor{logits.GetShape(), logits.GetData()}, outTensor{act.GetShape(), act.GetData()})
}

// execute runs the graph, terminating it if ctx ends first. The watcher is always joined before
// the run options are destroyed.
func (s *Session) execute(ctx context.Context, inputs, outputs []ort.Value) error {
	done := ctx.Done()
	if done == nil {
		return s.s.Run(inputs, outputs)
	}
	ro, err := ort.NewRunOptions()
	if err != nil {
		return fmt.Errorf("run options: %w", err)
	}
	defer ro.Destroy()
	stop := make(chan struct{})
	joined := make(chan struct{})
	go func() {
		defer close(joined)
		select {
		case <-done:
			ro.Terminate()
		case <-stop:
		}
	}()
	defer func() {
		close(stop)
		<-joined
	}()
	return s.s.RunWithOptions(inputs, outputs, ro)
}
