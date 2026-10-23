// Package router decides which Laya checkpoint should read a request. It is the routing half of
// laya/router.py: Route and RouteBatch look at the state, the questions and the caller's hints and
// name a checkpoint and the reason, without loading or running anything. Loading, hooks and
// routed inference are separate changes.
//
// The checkpoints are English, Multilingual and TypedDecisions. The order of the evidence is
// fixed, first match wins:
//
//	ForModel > ForTask > typed-decisions workflow (opt-in) > ForLang > ForLangGuess
//	  > WithLangGuess on the Router > language and script detection > the default checkpoint
//
// The default is Multilingual, not English: text whose language cannot be identified takes it, as
// text with no letters does. WithDefault(English) restores the English default for a deployment
// that is mostly English.
//
// Reason strings are part of the contract. They quote with Python's repr, round percentages half
// to even and cut at 60 code points, so a decision here reads exactly as the Python router's.
//
// A Router is immutable once built and safe for concurrent use.
//
// # Differences from Python
//
// A name is a string: Python's str(name) of a non-string has no counterpart. Router.register and
// the custom checkpoints are not ported, so the Router knows the three built-in names. A hint
// callable is a LangHint returning "" to abstain. explain=True is not ported: lang.Analysis has no
// diagnostic field. Hooks (on_route) and the model lifecycle belong to later changes.
package router
