# Python HTTP client

A synchronous, standard-library client for a self-hosted `laya-serve`.
The application process does not install or import `laya`, PyTorch, Transformers,
or NumPy. Use the core `laya` package when inference must run in your own process.

## Install from this checkout

This distribution is prepared as `laya-client`; it has not been published to PyPI.
In the application environment, from the repository root:

```bash
python -m pip install ./sdk/python
```

This uses setuptools to build the package. The installed SDK has no runtime
dependencies and supports Python 3.10–3.13.

Run the server in a **separate environment** using the existing
[self-hosting instructions](https://github.com/NandhaKishorM/laya#self-hosting-http-server-jev-compatible).
Installing `laya[serve]` in the application environment would install inference
dependencies again.

## Predict

```python
import os
from laya_client import LayaClient, LayaAPIError, LayaTimeoutError

client = LayaClient(
    base_url="http://127.0.0.1:8000",
    api_key=os.environ.get("LAYA_API_KEY"),
    timeout=120.0,
)
questions = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {"billing": "Payments and refunds", "technical": "Product support"},
    },
}
try:
    result = client.predict({"message": "I was charged twice"}, questions)
    print(result["answers"]["team"]["choice"])
except LayaAPIError as error:
    print(error.status_code, error.detail)
except LayaTimeoutError:
    print("Stopped waiting; the server may still be processing the request.")
```

`predict(state, questions, *, model=None, task=None, lang=None, lang_guess=None,
max_len=None, head_max_len=None, min_confidence=None, timeout=None)` calls
`POST /v1/systemone`. State is text, a JSON object, or an array; questions use
the existing [HTTP contract](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md).
Choice, score and noul answers are returned as ordinary dictionaries. Types are
available in `laya_client.types`.

Controls left as `None` are omitted so server defaults remain in effect.
In particular, `model` is unset by default. Explicit model names, workflow names,
language hints and threshold-map keys are resolved by the server. The client
does not duplicate its model registry or impose its configurable request limits.
Question extensions such as `option_order` pass through unchanged.

Results retain server extensions and unknown JSON fields. Strict Jev responses
are accepted without Laya metadata. If a strict server omits `abstention`, this
does **not** mean a requested threshold passed. Use non-strict responses when
your application needs that information. Tournament probabilities may contain
zeros for eliminated options; the reported confidence still belongs to the
final round and is not recalculated by the client.

`client.health()` calls `GET /health`. Even a server with an API key can return
`{"status": "ok"}` for an anonymous or incorrectly authenticated probe. Liveness
does not prove authentication or that checkpoints have been loaded.

## Connection and error behavior

- URLs may include a reverse-proxy prefix, e.g. `https://host.example/laya`.
  Use ASCII URLs (percent-encode paths and use IDNA hostnames where necessary),
  without credentials, whitespace, query strings or fragments.
- The optional API key is a printable ASCII bearer token without whitespace.
  It is passed explicitly; the SDK does not read environment variables itself.
- TLS certificate verification and Python's standard proxy settings remain enabled.
  Redirects are refused: configure the final server URL.
- Calls block the calling thread. Timeout is in seconds, positive and finite,
  subject to the platform timeout limit. A per-call `None` inherits the client's
  120-second default. This is a blocking I/O timeout, **not** a total request
  deadline or a server cancellation mechanism.
- Requests are never automatically retried, including on 503 or timeout.
- There is no connection pool or open client session to close. Each response is
  closed before the method returns or raises.

All client errors inherit `LayaError`:

| Exception | Meaning |
| --- | --- |
| `LayaValidationError` | Invalid settings or a request rejected locally before sending |
| `LayaAPIError` | Non-2xx response; `status_code` and `detail` preserve HTTP diagnostics |
| `LayaConnectionError` | Connection, DNS, TLS or interrupted-response failure |
| `LayaTimeoutError` | A blocking operation timed out; also a `LayaConnectionError` |
| `LayaResponseError` | A 2xx response contained invalid JSON or required prediction fields |

HTTP error details can be strings, objects or arrays. Error bodies are read up to
64 KiB; bodies that cannot be decoded as JSON are returned as text. The SDK does not log requests or credentials;
applications should handle server diagnostics according to their own logging policy.

The first version exposes `predict` and `health`. It does not implement async,
batch, local routing, model loading or Python prediction hooks.

## Development and verification

From the repository root, in a build environment:

```bash
python -m pip install build setuptools wheel
python -m build --no-isolation sdk/python
python sdk/python/tests/verify_package.py sdk/python/dist
```

The verifier builds an isolated client environment, installs the wheel with
normal dependency resolution and no package index, then runs the standard-library
tests against that installed package. It also checks the sdist-built wheel.

For live integration, supply a Python with `laya[serve]` dependencies installed:

```bash
python sdk/python/tests/verify_package.py sdk/python/dist --server-python /path/to/server/python
```

A tiny offline checkpoint exercises actual HTTP inference. The client process
has only the SDK installed. This verifies transport and inference parity, not
pretrained model quality. CI exercises the current server and the 0.4.7 fixture;
version compatibility is determined by those checks, not by a version handshake.

## Publishing

Package ownership, versioning and PyPI publication are maintained separately
from the core distribution. Build and inspect the wheel/sdist before release.
The core release workflow is not changed by this SDK.
