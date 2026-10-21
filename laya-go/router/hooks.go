package router

// Hooks are laya/hooks.py for the one event a pure Router raises, on_route (on_load and on_evict
// arrive with loading). A hook is any value with an OnRoute method; the same value may also be an
// agent.Hook, as in Python, where one object implements several events. Each call of Route
// dispatches over: the process-wide agent.DefaultHooks (which take only an agent.Hook, so a
// hook that is both embeds agent.BaseHook), the Router's installed hooks, then the
// call's ForHooks, and a hook without OnRoute is skipped.
//
// The dispatch is agent.HookRun, the one the agent uses, so ordering, WithHookErrors,
// WithHookTimeout and panic recovery behave as there. Unlike Python's Router, whose hooks run
// concurrently unless hooks_concurrent=False, hooks run one at a time across all goroutines
// unless WithConcurrentHooks is given, as on the agent. A hook must not call Route on the
// Router that is running it while hooks are serial: Python's lock is re-entrant, this one is not.
//
// A hook may replace ctx.Decision; the replacement is what Route returns. Route does not check
// it: a Model that names no checkpoint is the hook's business, and the caller that loads it
// sees the error. A hook that overruns WithHookTimeout is abandoned, not stopped (see
// agent/hooks.go).

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"reflect"
	"sync"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// RouteHook is a hook that sees the routing decision.
type RouteHook interface {
	// OnRoute runs after the Router has decided. It may replace ctx.Decision.
	OnRoute(ctx *RouteContext)
}

// RouteContext is what an OnRoute hook sees of one Route call. The hooks of a call share it and
// run one after another, so a replacement made by one is seen by the next.
type RouteContext struct {
	// States holds the one state being routed; rewriting it changes nothing.
	States []any
	// Questions are the questions asked, as passed to Route.
	Questions schema.Questions
	// RunID is 32 hex characters, shared by every hook of the call. It is not the prediction's.
	RunID string
	// Decision is the routing decision; assign to replace it.
	Decision Decision
	// Router is the Router that is routing.
	Router *Router
}

// BaseHook implements RouteHook with a no-op, for embedding.
type BaseHook struct{}

// OnRoute does nothing.
func (BaseHook) OnRoute(*RouteContext) {}

// HookFuncs is a RouteHook made of a function; nil is a no-op. Use it as *HookFuncs: a value
// holds a func, which cannot be compared, so RemoveHook could never find it.
type HookFuncs struct {
	OnRouteFunc func(*RouteContext)
}

// OnRoute calls OnRouteFunc.
func (f *HookFuncs) OnRoute(c *RouteContext) {
	if f.OnRouteFunc != nil {
		f.OnRouteFunc(c)
	}
}

// hookState is the Router's hook registry and dispatch settings.
type hookState struct {
	mu        sync.RWMutex
	installed []any
	timeout   time.Duration
	raise     bool
	lock      *agent.HookLock
	warn      func(string)
}

type hookConfig struct {
	installed  []any
	timeout    time.Duration
	raise      bool
	concurrent bool
	warn       func(string)
}

func newHookState(c hookConfig) *hookState {
	h := &hookState{installed: append([]any(nil), c.installed...), timeout: c.timeout, raise: c.raise, warn: c.warn}
	if h.warn == nil {
		h.warn = func(msg string) { log.Print(msg) }
	}
	if !c.concurrent {
		h.lock = agent.NewHookLock()
	}
	return h
}

// checkHooks refuses a nil hook and one that implements no event, Python's "hooks entries must
// implement at least one of ...".
func checkHooks(hooks []any) error {
	for i, h := range hooks {
		if isNil(h) {
			return fmt.Errorf("router: hook %d is nil", i)
		}
		_, route := h.(RouteHook)
		_, predict := h.(agent.Hook)
		if !route && !predict {
			return fmt.Errorf("router: hook %d (%T) implements neither OnRoute nor agent.Hook", i, h)
		}
	}
	return nil
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// WithInstalledHooks installs hooks on the Router, as AddHook does after New.
func WithInstalledHooks(hooks ...any) Option {
	return func(c *config) error {
		if err := checkHooks(hooks); err != nil {
			return err
		}
		c.hooks.installed = append(c.hooks.installed, hooks...)
		return nil
	}
}

// WithHookTimeout bounds the wait for each hook. d must be positive; leave the option out for no
// limit.
func WithHookTimeout(d time.Duration) Option {
	return func(c *config) error {
		if err := agent.ValidateHookTimeout(d); err != nil {
			return err
		}
		c.hooks.timeout = d
		return nil
	}
}

// WithHookErrors chooses what a failing hook does: raise true (the default) makes Route return
// its error, false warns and goes on.
func WithHookErrors(raise bool) Option {
	return func(c *config) error { c.hooks.raise = raise; return nil }
}

// WithConcurrentHooks lets hooks of concurrent calls run at the same time; the default runs one
// at a time.
func WithConcurrentHooks() Option {
	return func(c *config) error { c.hooks.concurrent = true; return nil }
}

// WithHookWarnings sets where a hook failure that does not fail the call is reported; the
// default is the log package.
func WithHookWarnings(sink func(msg string)) Option {
	return func(c *config) error { c.hooks.warn = sink; return nil }
}

// AddHook installs hooks after those already there. A hook added twice runs twice.
func (r *Router) AddHook(hooks ...any) error {
	if err := checkHooks(hooks); err != nil {
		return err
	}
	r.hk.mu.Lock()
	r.hk.installed = append(r.hk.installed, hooks...)
	r.hk.mu.Unlock()
	return nil
}

// RemoveHook removes every installed copy of hook, compared by identity, and reports whether
// there was one. A hook whose dynamic type cannot be compared is never found; pass a pointer.
func (r *Router) RemoveHook(hook any) bool {
	r.hk.mu.Lock()
	defer r.hk.mu.Unlock()
	kept := r.hk.installed[:0:0]
	for _, h := range r.hk.installed {
		if !agent.SameHook(h, hook) {
			kept = append(kept, h)
		}
	}
	removed := len(kept) != len(r.hk.installed)
	r.hk.installed = kept
	return removed
}

// InstallHooks installs hooks until the returned function is called, which removes one copy of
// each (the most recent) and leaves everything else, including what was added meanwhile.
// Overlapping installs may be undone in either order.
func (r *Router) InstallHooks(hooks ...any) (undo func(), err error) {
	if err := r.AddHook(hooks...); err != nil {
		return nil, err
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			r.hk.mu.Lock()
			defer r.hk.mu.Unlock()
			for _, h := range hooks {
				for i := len(r.hk.installed) - 1; i >= 0; i-- {
					if agent.SameHook(r.hk.installed[i], h) {
						r.hk.installed = append(r.hk.installed[:i:i], r.hk.installed[i+1:]...)
						break
					}
				}
			}
		})
	}, nil
}

// Hooks returns a copy of the installed hooks.
func (r *Router) Hooks() []any {
	r.hk.mu.RLock()
	defer r.hk.mu.RUnlock()
	return append([]any(nil), r.hk.installed...)
}

// composeHooks is the hooks of one call, in order: defaults, installed, the call's.
func (r *Router) composeHooks(c call) []any {
	var hooks []any
	for _, h := range agent.DefaultHooks() {
		hooks = append(hooks, h)
	}
	r.hk.mu.RLock()
	hooks = append(hooks, r.hk.installed...)
	r.hk.mu.RUnlock()
	return append(hooks, c.hooks...)
}

// ForHooks adds hooks to one request, after the defaults and the installed ones.
func ForHooks(hooks ...any) CallOption {
	return func(c *call) {
		if err := checkHooks(hooks); err != nil && c.err == nil {
			c.err = err
		}
		c.hooks = append(c.hooks, hooks...)
	}
}

// ForHookErrors overrides WithHookErrors for this request.
func ForHookErrors(raise bool) CallOption {
	return func(c *call) { c.raise = &raise }
}

// ForHookTimeout overrides WithHookTimeout for this request; d must be positive.
func ForHookTimeout(d time.Duration) CallOption {
	return func(c *call) { c.timeout = &d }
}

func newRunID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// dispatchRoute runs OnRoute over the hooks of c and returns the decision they leave.
func (r *Router) dispatchRoute(state any, qs schema.Questions, d Decision, c call) (Decision, error) {
	policy := agent.HookPolicy{Timeout: r.hk.timeout, Raise: r.hk.raise, Warn: r.hk.warn}
	if c.raise != nil {
		policy.Raise = *c.raise
	}
	if c.timeout != nil {
		if err := agent.ValidateHookTimeout(*c.timeout); err != nil {
			return d, err
		}
		policy.Timeout = *c.timeout
	}
	var hooks []RouteHook
	var all []any
	for _, h := range r.composeHooks(c) {
		if rh, ok := h.(RouteHook); ok {
			hooks, all = append(hooks, rh), append(all, h)
		}
	}
	if len(hooks) == 0 {
		return d, nil
	}
	rc := &RouteContext{States: []any{state}, Questions: qs, RunID: newRunID(), Decision: d, Router: r}
	run := agent.HookRun{Lock: r.hk.lock, Policy: policy}
	err := run.Dispatch(context.Background(), "on_route", len(hooks),
		func(i int) string { return agent.HookName(all[i]) }, func(i int) { hooks[i].OnRoute(rc) })
	if err != nil {
		return d, err
	}
	return rc.Decision, nil
}
