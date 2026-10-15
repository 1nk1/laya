package agent

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// gauge counts hooks running at once and remembers the peak.
type gauge struct{ now, peak atomic.Int32 }

func (g *gauge) enter() {
	n := g.now.Add(1)
	for {
		p := g.peak.Load()
		if n <= p || g.peak.CompareAndSwap(p, n) {
			return
		}
	}
}
func (g *gauge) leave() { g.now.Add(-1) }

func slowHook(g *gauge, d time.Duration) Hook {
	return &HookFuncs{OnStart: func(*PredictContext) { g.enter(); time.Sleep(d); g.leave() }}
}

func predictN(t *testing.T, a *Agent, n int, opts ...CallOption) {
	t.Helper()
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"), opts...); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

func TestSerialHooksNeverOverlap(t *testing.T) {
	g := &gauge{}
	a := testAgent(t, options{raise: true, installed: []Hook{slowHook(g, 2*time.Millisecond)}})
	predictN(t, a, 16)
	if p := g.peak.Load(); p != 1 {
		t.Fatalf("peak concurrent hooks = %d, want 1", p)
	}
}

func TestConcurrentHooksOverlap(t *testing.T) {
	g := &gauge{}
	a := testAgent(t, options{raise: true, concurrent: true, installed: []Hook{slowHook(g, 50*time.Millisecond)}})
	predictN(t, a, 8)
	if p := g.peak.Load(); p < 2 {
		t.Fatalf("peak concurrent hooks = %d, want at least 2", p)
	}
}

func TestTimeoutReturnsWhileHookRuns(t *testing.T) {
	release, started, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	h := &HookFuncs{OnStart: func(*PredictContext) { close(started); <-release; close(finished) }}
	a := testAgent(t, options{raise: true, timeout: 20 * time.Millisecond, installed: []Hook{h}})
	t0 := time.Now()
	_, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"))
	if err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("err = %v, want a timeout", err)
	}
	if d := time.Since(t0); d > 2*time.Second {
		t.Fatalf("returned after %v", d)
	}
	<-started
	select {
	case <-finished:
		t.Fatal("the hook was stopped; Go cannot, and the contract is that it keeps running")
	default:
	}
	close(release)
	<-finished
}

func TestPanicIsAnError(t *testing.T) {
	h := &HookFuncs{OnStart: func(*PredictContext) { panic("kaboom") }}
	a := testAgent(t, options{raise: true, installed: []Hook{h}})
	_, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"))
	if err == nil || !strings.Contains(err.Error(), "kaboom") {
		t.Fatalf("err = %v", err)
	}
}

func TestRaiseErrorsFalseSwallowsAHookError(t *testing.T) {
	var onErr atomic.Int32
	var warns []string
	var mu sync.Mutex
	h := &HookFuncs{
		OnStart: func(*PredictContext) { panic("nope") },
		OnErr:   func(c *PredictContext) { onErr.Add(1) },
	}
	a := testAgent(t, options{raise: false, installed: []Hook{h}, warn: func(s string) { mu.Lock(); warns = append(warns, s); mu.Unlock() }})
	res, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"))
	if err != nil || len(res) != 1 {
		t.Fatalf("res = %v, err = %v", res, err)
	}
	// Python's start hook failing is not an inference failure: on_error runs only for the latter.
	if len(warns) != 1 || !strings.Contains(warns[0], "nope") {
		t.Fatalf("warnings = %q", warns)
	}
	if n := onErr.Load(); n != 0 {
		t.Fatalf("on_error ran %d times for a start hook that only warned", n)
	}
}

func TestOnErrorRunsWhenInferenceFailsAndRaiseIsFalse(t *testing.T) {
	var seen atomic.Value
	h := &HookFuncs{OnErr: func(c *PredictContext) { seen.Store(c.Err.Error()) }}
	a := testAgent(t, options{raise: false, installed: []Hook{h}, warn: func(string) {}})
	a.hk.stub = func(context.Context, []any, schema.Questions, call) ([]schema.Prediction, error) {
		return nil, errors.New("the graph refused this batch")
	}
	_, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"))
	if err == nil {
		t.Fatal("inference failed but the call succeeded: raise_errors only governs hook errors")
	}
	if s, _ := seen.Load().(string); s != "the graph refused this batch" {
		t.Fatalf("on_error saw %q", s)
	}
}

func TestCloseDuringRunningHookReturns(t *testing.T) {
	release, started := make(chan struct{}), make(chan struct{})
	h := &HookFuncs{OnStart: func(*PredictContext) { close(started); <-release }}
	a := testAgent(t, options{raise: true, installed: []Hook{h}})
	done := make(chan error, 1)
	go func() {
		_, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"))
		done <- err
	}()
	<-started
	closed := make(chan struct{})
	go func() { a.Close(); close(closed) }()
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("Close deadlocked behind a running hook")
	}
	if err := <-done; !errors.Is(err, ErrClosed) {
		t.Fatalf("call err = %v, want ErrClosed", err)
	}
	close(release)
}

func TestValidateTimeout(t *testing.T) {
	for _, bad := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if err := validateTimeout(bad); err == nil || !strings.HasPrefix(err.Error(), "hooks_timeout must be a positive finite number") {
			t.Errorf("validateTimeout(%v) = %v", bad, err)
		}
	}
	if err := validateTimeout(0.05); err != nil {
		t.Error(err)
	}
}

func TestPredictLongRefusesHooks(t *testing.T) {
	a := testAgent(t, options{raise: true})
	_, err := a.PredictLong(context.Background(), "s", questionsOf("q"), WithHooks(&StubHook{name: "h", rec: &recorder{}}))
	if err == nil || !strings.Contains(err.Error(), "PredictLong runs no hooks") {
		t.Fatalf("err = %v", err)
	}
}

func TestNilHookRefused(t *testing.T) {
	a := testAgent(t, options{raise: true})
	if err := a.AddHook(nil); err == nil {
		t.Error("AddHook(nil) accepted")
	}
	if _, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"), WithHooks(nil)); err == nil {
		t.Error("WithHooks(nil) accepted")
	}
}

func TestOpenOptionError(t *testing.T) {
	var o options
	WithHookTimeout(-time.Second)(&o)
	if o.err == nil {
		t.Fatal("a negative timeout was accepted")
	}
}

func TestZeroAgentHookMethodsDoNotPanic(t *testing.T) {
	var a Agent
	if err := a.AddHook(&StubHook{name: "h", rec: &recorder{}}); err == nil {
		t.Error("AddHook on a zero Agent was accepted")
	}
	if _, err := a.InstallHooks(&StubHook{name: "h", rec: &recorder{}}); err == nil {
		t.Error("InstallHooks on a zero Agent was accepted")
	}
	if a.RemoveHook(&StubHook{}) || a.Hooks() != nil {
		t.Error("zero Agent reports hooks")
	}
}
