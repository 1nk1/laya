// Package schema holds the values every part of laya-go exchanges: the questions put to the
// model (Question, Questions) and what comes back (Answer, Prediction, Usage). It is the Go form
// of the question dicts and result dicts of laya/agent.py and laya/common.py.
//
// It is a leaf: it imports only lang (for lang.Object, the order-keeping mapping a criterion may
// be) and never cgo, so the ONNX backend, the router, the shortlist and the presets can all share
// these types without import cycles, and a caller can build and validate questions without
// linking a runtime.
//
// # Questions
//
// A Question is immutable and valid by construction: Choice, ChoiceOf, Score, Noul and NoulWith
// refuse what laya's Agent._check_question refuses, with the same reasons. Three rules decide
// what the model is asked, and a builder must keep them:
//
//   - Option order is part of the question. A choice's options are rendered in the order they
//     are written, the Nth option's probability is reported against the Nth label, and two
//     orders are two different questions. That is why options are a variadic list and never a
//     Go map, whose order is random.
//   - A noul's options are always [false, true], in that order, whatever the labels say.
//   - A criterion is a string, rendered as written, or a structured value (bool, number,
//     json.Number, slice, map[string]any or lang.Object), rendered as
//     json.dumps(value, ensure_ascii=False) spells it. A nil or "" criterion means "no
//     description"; 0 and false are descriptions. A value json.dumps would not write as JSON, a
//     struct or a channel, is refused.
//   - A choice label is a string. Python also takes number and bool labels and renders them with
//     str(); spell them that way here ("1", "2.5", "True"). ParseQuestions does it for a JSON
//     list of labels, and refuses two labels that spell the same, which Python keeps apart as 1
//     and "1" and an answer keyed by text cannot.
//
// Questions is the ordered set a request asks, keyed by question id; answers come back in the
// same order. ParseQuestions reads the Python wire form ({"type", "instructions", "criteria",
// "labels", "option_order"}) in document order.
//
// # Answers
//
// Every number in an Answer is rounded to four decimals as Python's round(x, 4) rounds them
// (half to even on the exact binary value). AnswerConfidence and Confidence are different
// quantities on different scales and must not be compared against one threshold; see Answer.
//
// Every value here is safe for concurrent reads.
package schema
