# Agent

`laya.Agent` loads one checkpoint and answers typed questions about a state. `laya.load` is
a shortcut for `Agent(...)` for every backend except `backend="onnx"`, which returns an
`ONNXAgent` instead, and `laya.RLAgent` is an alias of `Agent`. `ONNXAgent` runs an
exported ONNX model on CPU; import it from `laya.onnx_agent`. It is not an `Agent` subclass --
both are `HookRegistry`s, but only `Agent` holds a PyTorch model, so
`isinstance(laya.load(..., backend="onnx"), laya.Agent)` is `False`.

::: laya.agent.Agent

::: laya.agent.load

::: laya.onnx_agent.ONNXAgent

## Quantized export

`scripts/export_onnx.py --quantize` writes an INT8 weight-only quantized copy beside the fp32
export (`laya.onnx` also produces `laya.int8.onnx`). Dynamic quantization converts the `MatMul`
weights to int8 with the activation scale computed per input at run time, so no calibration
dataset is needed, and `ONNXAgent` loads the result by pointing `onnx_path` at it. On CPU it is
roughly 2x faster than the eager model and ~1.8x faster than the fp32 ONNX graph, and 1.4-2.8x
smaller depending on the checkpoint.

INT8 trades real accuracy, so it is a size/latency option, not a free one — do not use it where
the calibrated probability or confidence matters. Scales are **per-tensor** by default; `--per-channel`
opts into per-channel weights but on the dynamic path that collapses the decision model (agreement
with the eager model dropped to ~32% on the English checkpoint and ~40% on the multilingual one,
vs ~67% / ~83% per-tensor; see issue #790). Even per-tensor drifts noticeably on the larger
checkpoint; accuracy-safe int8 would need QAT or SmoothQuant-style outlier handling. The int8
graph is CPU-only: ONNX Runtime has no INT8 MatMul kernel on the CUDAExecutionProvider, and a GPU
provider silently falls back per node.

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
```

## In-process dynamic quantization

The quantized export above is the int8 path `ONNXAgent` loads. `torch.ao.quantization.quantize_dynamic`
on a live `Agent` is a different operation.

[Issue #1065](https://github.com/NandhaKishorM/laya/issues/1065) reports that the default fbgemm
engine crashes with SIGILL (illegal instruction) on a CPU with SSE4.2 and no AVX2 (laya 0.3.22,
torch 2.14). Check the CPU first. `torch.backends.cpu.get_cpu_capability()` returns `"NO AVX"`
when AVX2 is unavailable, and `"AVX2"` or `"AVX512"` when it is. On Linux,
`grep -m1 flags /proc/cpuinfo` prints the raw flags (`avx2` is absent in the report). The
setting that ran there is qnnpack, applied only to the encoder:

```python
import torch
import torch.nn as nn

torch.backends.quantized.engine = "qnnpack"
torch.ao.quantization.quantize_dynamic(
    agent.model.encoder, {nn.Linear}, dtype=torch.qint8, inplace=True)
```

Encoder-only quantization does run, with either engine the CPU can execute. It is not a speedup
you can assume, and it changes scores. On the machine in #1065, qnnpack took 3.7 s against 3.0 s
in fp32 for 20 states, and one candidate's `noul` went from 0.15 to 0.04. A spot check on the
multilingual checkpoint with torch 2.14.1 (capability `AVX512`, four threads, short states) saw
the same kind of move: qnnpack took 2.1 s against 0.57 s, and one `noul` went from 0.90 to 0.13.
fbgemm was faster on that CPU (0.39 s) and still moved the score, to 0.15. Validate latency and
probabilities on your own data.

Do not quantize the whole `agent.model`. That replaces the decision head's `nn.Linear` layers
with dynamically quantized linears, whose `weight` is a method rather than a tensor, and two
different reads of that method can be the one that raises.

On every published checkpoint it is the encoder stack. `agent.model.head` is an
`nn.TransformerEncoder` of `head_layers` `nn.TransformerEncoderLayer`s
(`laya/common.py::DecisionModel.__init__`, and `None` when `head_layers` is 0), and each layer's
sparsity fast path reads `.device` on the quantized weights, so `predict` raises
`AttributeError: 'function' object has no attribute 'device'`. Reproduced on `laya`,
`laya-multilingual` and `laya-typed-decisions`, qnnpack engine, from
`torch/nn/modules/transformer.py`'s `(x.device.type in _supported_device_type)` line.

Whether that fast path is taken is what decides which message you get, so a reader who sees the
other one has not found a different bug. Torch skips the fast path when `self_attn.num_heads` is
odd, and the head takes `nhead = max(1, d // 64)`, so a model built from a config whose hidden
size is under 128 gets `nhead = 1`, survives the stack, and fails one step later in
`DecisionModel.forward`, which reads `.dtype` on `act_head[0].weight` whenever autocast is off:
`AttributeError: 'function' object has no attribute 'dtype'`. The published checkpoints are
`d=768` and `d=1024`, so `nhead` is 12 or 16 and never that case.

Either way, leave the head in fp32.
