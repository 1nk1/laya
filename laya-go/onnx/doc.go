// Package onnx runs a laya graph on ONNX Runtime and implements infer.Session, so the pure-Go
// agent package can answer questions with it.
//
// It is a separate module because it needs cgo and a native ONNX Runtime shared library
// (1.29 or newer, which the caller supplies). Open takes the fused laya.onnx, whose .data
// sidecar must sit beside it, and runs it on the CPU provider:
//
//	sess, err := onnx.Open("laya.onnx", onnx.WithLibrary("/opt/onnxruntime/libonnxruntime.so.1.30.0"))
//	a, err := agent.Open(os.DirFS("checkpoint"), sess)
//
// The library is found through WithLibrary, else $LAYA_ONNXRUNTIME_LIBRARY, else the platform's
// default name on the loader's search path. A process holds one ONNX Runtime: sessions share it,
// and it is unloaded when the last one closes. Built with CGO_ENABLED=0 the package still
// compiles, and Open returns ErrNoCGO.
package onnx
