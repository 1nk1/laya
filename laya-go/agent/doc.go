// Package agent answers typed questions about a state with a laya checkpoint: the port of
// ONNXAgent.predict and predict_batch. It is pure Go. The model runs behind an infer.Session,
// which a native backend supplies, so this package links no runtime.
//
//	a, err := agent.Open(os.DirFS("checkpoints/english"), session)
//	pred, err := a.Predict(ctx, "I was charged twice", questions)
//
// A call encodes the state once, builds one row per question, collates the rows of up to
// WithBatchSize states into one batch, runs it, and decodes the logits into the typed answers
// and the usage report of laya's predict().
//
// # Order and batching
//
// Answers follow the question order and results follow the input order, whatever the batch size
// or sorting. Batch shape does change padding, and with it a probability's last digits, so
// results are compared with a tolerance across shapes, not byte for byte.
//
// # Differences from Python
//
// With no questions, Python writes a usage of two keys (input_tokens, output_tokens); schema.Usage
// writes all seven, and so does this package.
//
// A nil state is refused, with or without questions, where Python raises a TypeError only when
// there are questions. A nil slice or map is an empty list or dict, as in Python.
//
// max_len and head_max_len are checked at load and must be positive integers; Python reads them
// with cfg.get and checks nothing until a call uses them.
//
// A temperature that is not a list of exactly 3 numbers is refused at load, as Python's torch
// Agent refuses it; ONNXAgent loads it.
//
// A NaN or infinite logit is refused, including a -Inf for a real option, where Python would
// answer probability 0 for that option. A model that emits one is broken.
//
// # Scope
//
// Not here yet: per-language temperature overrides, the calibration file, min_confidence,
// hooks, predict_long, digest checks and per-call max_len overrides. WithLanguage is accepted
// so callers need not change when overrides arrive; without overrides it does nothing.
package agent
