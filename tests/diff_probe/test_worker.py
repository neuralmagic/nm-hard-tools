from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nm_hard_tools.diff_probe import worker
from nm_hard_tools.diff_probe.models import ProbeReport
from nm_hard_tools.diff_probe.worker import (
    ProbeFailure,
    divergence,
    is_degenerate,
    load_request_set,
    repeated_tail,
    smoke_checks,
    summarize_diff,
    summarize_smoke,
)

# --- pure measurement logic ---------------------------------------------------


def test_repeated_tail_catches_classic_garbage() -> None:
    assert repeated_tail("The image shows shows shows shows shows shows")
    assert repeated_tail("ok " + "na" * 40)
    assert not repeated_tail("The image shows a red bicycle next to a wall.")


def test_degeneracy_flags_empty_repetitive_and_single_symbol() -> None:
    assert is_degenerate("")
    assert is_degenerate("   \n ")
    assert is_degenerate("!!!" * 40)
    assert is_degenerate("word " * 60)
    assert not is_degenerate("A concise, correct description of the scene.")


def test_divergence_is_zero_for_identical_and_high_for_garbage() -> None:
    assert divergence("same text", "same text") == 0.0
    healthy = "The chart shows quarterly revenue rising."
    garbage = "the the the the the the the the the the"
    assert divergence(healthy, garbage) > 0.5


def test_summarize_diff_reads_a_broken_side() -> None:
    pairs = [
        ("shows shows shows shows shows", "A cat on a mat."),
        ("!!!" * 50, "Two people at a table."),
        ("A red bicycle.", "A red bicycle."),
    ]
    metrics = summarize_diff(pairs)
    assert metrics["request_count"] == 3.0
    assert metrics["mean_divergence"] > 0.5
    assert metrics["exact_match_rate"] == pytest.approx(1 / 3)
    assert metrics["degenerate_rate_a"] == pytest.approx(2 / 3)
    assert metrics["degenerate_rate_b"] == 0.0


def test_summarize_diff_reads_a_healthy_pair() -> None:
    pairs = [("A red bicycle.", "A red bicycle."), ("Two cats.", "Two cats.")]
    metrics = summarize_diff(pairs)
    assert metrics["mean_divergence"] == 0.0
    assert metrics["exact_match_rate"] == 1.0


def test_smoke_checks() -> None:
    assert smoke_checks("A fine answer.", "stop") == (True, False)
    assert smoke_checks("", "stop") == (False, True)
    assert smoke_checks("A fine answer.", None) == (False, False)
    passed, degenerate = smoke_checks("no no no no no no no no", "length")
    assert not passed and degenerate


def test_summarize_smoke_counts() -> None:
    rows = [
        {"text": "fine", "checks_passed": True, "degenerate": False},
        {"text": "", "checks_passed": False, "degenerate": True},
    ]
    metrics = summarize_smoke(rows)
    assert metrics["pass_rate"] == 0.5
    assert metrics["empty_rate"] == 0.5


# --- request-set freezing -----------------------------------------------------


def _write_set(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_request_set_hash_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    digest = _write_set(path, [{"messages": [{"role": "user", "content": "hi"}]}])
    assert load_request_set(path, digest, "chat", None, 64)
    path.write_bytes(path.read_bytes() + b'{"messages": []}\n')
    with pytest.raises(ProbeFailure, match="does not match the pinned"):
        load_request_set(path, digest, "chat", None, 64)


def test_request_set_shape_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    digest = _write_set(path, [{"prompt": "hello"}])
    with pytest.raises(ProbeFailure, match="no messages list"):
        load_request_set(path, digest, "chat", None, 64)
    assert load_request_set(path, digest, "completions", None, 64)


def test_request_set_limit_and_bounds(tmp_path: Path) -> None:
    rows = [{"prompt": f"p{i}"} for i in range(5)]
    path = tmp_path / "set.jsonl"
    digest = _write_set(path, rows)
    assert len(load_request_set(path, digest, "completions", 2, 64)) == 2
    with pytest.raises(ProbeFailure, match="over the configured maximum"):
        load_request_set(path, digest, "completions", None, 3)


# --- worker run against stub endpoints ---------------------------------------


class StubEndpoint:
    answers: ClassVar[dict[str, list[str]]] = {}
    calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, spec: dict[str, Any], timeout_seconds: float) -> None:
        self.spec = spec
        self.role = spec.get("role_answers", "a")

    def close(self) -> None:
        pass

    def generate(
        self, api: str, model: str, row: dict[str, Any], max_tokens: int
    ) -> tuple[str, str | None]:
        StubEndpoint.calls.append((self.spec["identity"], api))
        queue = StubEndpoint.answers[self.spec["identity"]]
        return queue.pop(0), "stop"


def worker_config(tmp_path: Path, mode: str) -> dict[str, Any]:
    rows = [
        {"id": "r1", "messages": [{"role": "user", "content": "Describe."}]},
        {"id": "r2", "messages": [{"role": "user", "content": "Explain."}]},
    ]
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    request_sets = tmp_path / "sets"
    request_sets.mkdir(exist_ok=True)
    (request_sets / "vision.jsonl").write_bytes(payload)
    endpoints = {
        "a": {
            "logical_name": "a",
            "url": "http://a/v1/chat/completions",
            "identity": "http://a/v1/chat/completions",
            "tls_verify": True,
            "api_key_env": "OPENAI_API_KEY_A",
        },
    }
    request: dict[str, Any] = {
        "mode": mode,
        "target_a": "a",
        "target_b": None,
        "model": "m",
        "request_set": "vision",
        "limit": None,
        "max_tokens": None,
        "timeout_seconds": 600,
    }
    if mode == "diff":
        endpoints["b"] = {
            "logical_name": "b",
            "url": "http://b/v1/chat/completions",
            "identity": "http://b/v1/chat/completions",
            "tls_verify": True,
            "api_key_env": "OPENAI_API_KEY_B",
        }
        request["target_b"] = "b"
    return {
        "probe_id": "probe-" + "1" * 20,
        "request": request,
        "request_set": {
            "file": "vision.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "api": "chat",
            "max_requests": 64,
            "default_max_tokens": 128,
        },
        "endpoints": endpoints,
        "max_tokens": 128,
        "effective_configuration": {"probe_id": "probe-" + "1" * 20},
    }


def test_diff_run_produces_report_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "Endpoint", StubEndpoint)
    StubEndpoint.answers = {
        "http://a/v1/chat/completions": ["shows shows shows shows shows", "fine"],
        "http://b/v1/chat/completions": ["A cat.", "fine"],
    }
    config = worker_config(tmp_path, "diff")
    report = worker.run(config, tmp_path / "results", tmp_path / "sets")
    assert report.terminal_state == "succeeded"
    assert report.primary_metric == "mean_divergence"
    assert report.metrics["mean_divergence"] > 0.2
    assert report.metrics["degenerate_rate_a"] == 0.5
    output = tmp_path / "results" / config["probe_id"]
    samples = (output / "samples.jsonl").read_text().splitlines()
    assert len(samples) == 2
    persisted = ProbeReport.model_validate(
        json.loads((output / "report.json").read_text())
    )
    assert persisted.metrics == report.metrics
    manifest = json.loads((output / "artifacts.json").read_text())
    assert {a["name"] for a in manifest["artifacts"]} == {
        "effective-configuration.json",
        "samples.jsonl",
        "report.json",
    }
    assert all(len(a["sha256"]) == 64 for a in manifest["artifacts"])


def test_smoke_run_scores_pass_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "Endpoint", StubEndpoint)
    StubEndpoint.answers = {
        "http://a/v1/chat/completions": ["A fine answer.", ""],
    }
    config = worker_config(tmp_path, "smoke")
    report = worker.run(config, tmp_path / "results", tmp_path / "sets")
    assert report.primary_metric == "pass_rate"
    assert report.metrics["pass_rate"] == 0.5


def test_transport_failure_fails_the_probe_with_a_durable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenEndpoint(StubEndpoint):
        def generate(
            self, api: str, model: str, row: dict[str, Any], max_tokens: int
        ) -> tuple[str, str | None]:
            raise ProbeFailure("request to http://a failed after 3 attempts")

    monkeypatch.setattr(worker, "Endpoint", BrokenEndpoint)
    config = worker_config(tmp_path, "smoke")
    with pytest.raises(ProbeFailure):
        worker.run(config, tmp_path / "results", tmp_path / "sets")
    worker._write_failure(
        config,
        tmp_path / "results",
        ProbeFailure("request to http://a failed after 3 attempts"),
        worker._timestamp(),
        1.0,
    )
    output = tmp_path / "results" / config["probe_id"]
    report = ProbeReport.model_validate(
        json.loads((output / "report.json").read_text())
    )
    assert report.terminal_state == "failed"
    assert "failed after 3 attempts" in (report.error or "")
    assert (output / "error.txt").exists()
