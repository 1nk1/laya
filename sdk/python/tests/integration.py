"""Run the installed SDK against a separate Python process with real offline inference."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

from laya_client import LayaAPIError, LayaClient, LayaConnectionError


def run(server_python, fixture_path, strict=False, legacy=False):
    with tempfile.TemporaryDirectory(prefix="laya-sdk-live-") as directory:
        root = Path(directory)
        command = [str(server_python), "-u", str(fixture_path)]
        if not legacy:
            command.append("--tournament")
            if strict:
                command.append("--strict")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_TOURNAMENT_GROUP="0",
                   LAYA_JEV_STRICT="0", LAYA_PRELOAD="0", LAYA_IDLE_UNLOAD_SECONDS="0")
        with (root / "stderr.txt").open("w+", encoding="utf-8", errors="replace") as stderr:
            server = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                      stderr=stderr, text=True, encoding="utf-8")
            lines = queue.Queue()

            def read_lines():
                for line in server.stdout:
                    if line.startswith("LAYA_TEST_SERVER="):
                        lines.put(json.loads(line.split("=", 1)[1]))
                        return
                lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            try:
                fixture = lines.get(timeout=90)
                if fixture is None:
                    raise RuntimeError("Server exited before producing its fixture")
                client = LayaClient(base_url=fixture["baseURL"], api_key="integration-test", timeout=10)
                deadline = time.monotonic() + 15
                while True:
                    try:
                        client.health(timeout=0.5)
                        break
                    except LayaConnectionError:
                        if time.monotonic() >= deadline or server.poll() is not None:
                            raise
                        time.sleep(0.05)
                result = client.predict(fixture["state"], fixture["questions"], model="english")
                expected = fixture["expected"]
                assert result["model"] == expected["model"]
                if strict:
                    fields = {"choice": ("type", "choice", "probabilities", "confidence"),
                              "score": ("type", "score", "probabilities", "confidence", "legend"),
                              "noul": ("type", "noul")}
                    projected = {
                        "model": expected["model"],
                        "answers": {qid: {k: answer[k] for k in fields[answer["type"]]}
                                    for qid, answer in expected["answers"].items()},
                        "usage": {k: expected["usage"][k] for k in ("input_tokens", "output_tokens")},
                    }
                    assert result == projected, (result, projected)
                    gated = client.predict(fixture["state"], fixture["questions"], min_confidence=0.99)
                    assert all("abstention" not in a for a in gated["answers"].values())
                else:
                    for qid, answer in expected["answers"].items():
                        for key, value in answer.items():
                            assert result["answers"][qid][key] == value, (qid, key)
                    for key, value in expected["usage"].items():
                        assert result["usage"][key] == value, key
                    assert result["routing"]["model"] == "english"
                    auto = client.predict(fixture["state"], fixture["questions"])
                    assert auto["routing"]["model"] == "multilingual"
                    gated = client.predict(fixture["state"], fixture["questions"], lang="en",
                                           max_len=128, head_max_len=64, min_confidence=0.0)
                    assert gated["routing"]["model"] == "english"
                    assert all(a["abstention"] == "passed" for a in gated["answers"].values())

                if not legacy:
                    labels = {str(i): "Option %d" % i for i in range(5)}
                    answer = client.predict("hello", {"large": {
                        "type": "choice", "instructions": "Pick one", "criteria": labels,
                    }})["answers"]["large"]
                    assert set(answer["probabilities"]) == set(labels)
                    assert answer["choice"] in labels
                    if strict:
                        assert "x_tournament" not in answer
                    else:
                        finalists = answer["x_tournament"]["finalists"]
                        assert 0 < len(finalists) <= 2
                        assert all(p == 0.0 for k, p in answer["probabilities"].items() if k not in finalists)

                anonymous = LayaClient(base_url=fixture["baseURL"], api_key="wrong")
                assert anonymous.health() == {"status": "ok"}
                try:
                    anonymous.predict("hello", fixture["questions"])
                except LayaAPIError as error:
                    assert error.status_code == 401
                else:
                    raise AssertionError("Unauthenticated prediction succeeded")
                try:
                    client.predict("x" * 50_001, fixture["questions"])
                except LayaAPIError as error:
                    assert error.status_code == 413
                else:
                    raise AssertionError("Server request limit was bypassed")
                print("PASS real HTTP inference, controls, auth and limits: "
                      + ("legacy fixture" if legacy else "strict" if strict else "extended/tournament"), flush=True)
            except BaseException:
                stderr.flush()
                stderr.seek(0)
                print(stderr.read(), file=sys.stderr)
                raise
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                server.stdout.close()
                reader.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-python", required=True, type=Path)
    parser.add_argument("--server-fixture", type=Path,
                        default=Path(__file__).resolve().parents[3] / "tests" / "sdk_server_fixture.py")
    parser.add_argument("--legacy-fixture", action="store_true")
    args = parser.parse_args()
    run(args.server_python.resolve(), args.server_fixture.resolve(), legacy=args.legacy_fixture)
    if not args.legacy_fixture:
        run(args.server_python.resolve(), args.server_fixture.resolve(), strict=True)
