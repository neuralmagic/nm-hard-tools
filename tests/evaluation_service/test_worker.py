from __future__ import annotations

import importlib.resources
import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from lm_eval.tasks._yaml_loader import load_yaml

from nm_hard_tools.evaluation_service import worker
from nm_hard_tools.evaluation_service.worker import (
    BoundedTee,
    LogBudget,
    run,
    summarize_metrics,
)

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k_small.jsonl"
GSM8K_TASK = {
    "name": "gsm8k",
    "config_path": "lm_eval/tasks/gsm8k/gsm8k.yaml",
    "dataset": {
        "path": "openai/gsm8k",
        "name": "main",
        "revision": DATASET_REVISION,
    },
}


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        size = int(self.headers["content-length"])
        payload = json.loads(self.rfile.read(size))
        prompts = payload["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        choices = []
        for index, prompt in enumerate(prompts):
            if "buys 3" in prompt:
                text = "We add them. #### 5"
            elif "loses 4" in prompt:
                text = "Subtracting gives #### 5"
            else:
                text = "I cannot extract a number."
            choices.append({"index": index, "text": text})
        body = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def test_production_task_loader_uses_installed_lm_eval_package() -> None:
    task = worker._task_config(GSM8K_TASK)
    assert task["task"] == "gsm8k"
    assert task["dataset_path"] == "openai/gsm8k"
    assert task["dataset_name"] == "main"
    assert task["dataset_kwargs"]["revision"] == DATASET_REVISION


def test_deterministic_fixture_and_fake_openai_endpoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/completions"

    task = load_yaml(
        importlib.resources.files("lm_eval") / "tasks" / "gsm8k" / "gsm8k.yaml",
        resolve_func=True,
    )
    task.pop("dataset_name", None)
    task["dataset_path"] = "json"
    task["dataset_kwargs"] = {
        "data_files": {"train": str(FIXTURE), "test": str(FIXTURE)}
    }
    monkeypatch.setattr(worker, "_task_config", lambda _spec: task)
    config = {
        "evaluation_id": "eval-" + "1" * 20,
        "request": {
            "target": "fixture",
            "model": "fixture",
            "profile": "gsm8k",
            "num_fewshot": 0,
            "limit": 3,
            "generation_limit": 64,
            "timeout_seconds": 60,
        },
        "endpoint_identity": endpoint,
        "effective_configuration": {"profile": "gsm8k", "fixture": "pinned"},
        "invocation": {
            "model_backend": "local-completions",
            "model_args": {
                "model": "fixture",
                "base_url": endpoint,
                "tokenizer_backend": "none",
                "tokenized_requests": False,
                "num_concurrent": 1,
                "max_retries": 0,
                "max_gen_toks": 64,
                "timeout": 10,
                "verify_certificate": True,
                "seed": 0,
            },
            "tasks": [GSM8K_TASK],
            "primary_metric": {
                "task": "gsm8k",
                "metric": "exact_match",
                "filter": "flexible-extract",
            },
            "num_fewshot": 0,
            "batch_size": 1,
            "limit": 3,
            "bootstrap_iters": 0,
            "log_samples": True,
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "generation_kwargs": {
                "max_gen_toks": 64,
                "do_sample": False,
                "temperature": 0.0,
            },
            "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
            "confirm_run_unsafe_code": False,
            "cache_requests": False,
            "rewrite_requests_cache": False,
            "delete_requests_cache": False,
            "check_integrity": False,
            "write_out": False,
            "predict_only": False,
            "system_instruction": None,
            "metadata": {},
        },
    }
    try:
        report = run(config, tmp_path)
    finally:
        server.shutdown()
        thread.join()
    assert report.primary_metric is not None
    assert report.primary_metric.value == 2 / 3
    assert report.sample_counts == {"gsm8k": 3}
    identity = f"openai/gsm8k:main@{DATASET_REVISION}"
    assert report.dataset_fingerprints[identity]["test"]
    output = tmp_path / config["evaluation_id"]
    # lm-eval logs one sample per configured GSM8K filter.
    assert (output / "samples.jsonl").read_text().count("\n") == 6
    assert json.loads((output / "artifacts.json").read_text())["artifacts"]


def test_summary_preserves_generic_metrics_and_primary_value() -> None:
    result = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.5,
                "exact_match,flexible-extract": 1.0,
                "exact_match_stderr,flexible-extract": 0.0,
                "alias": "gsm8k",
            }
        },
        "n-samples": {"gsm8k": {"effective": 3}},
    }
    metrics, primary, counts = summarize_metrics(
        result,
        {
            "task": "gsm8k",
            "metric": "exact_match",
            "filter": "flexible-extract",
        },
    )
    assert metrics["gsm8k"]["exact_match,strict-match"] == 0.5
    assert primary.value == 1.0
    assert counts == {"gsm8k": 3}


def test_missing_primary_metric_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="primary metric"):
        summarize_metrics(
            {"results": {"task": {}}, "n-samples": {}},
            {"task": "task", "metric": "acc", "filter": "none"},
        )


def test_stdout_and_stderr_share_one_log_budget(tmp_path: Path) -> None:
    class Sink:
        encoding = "utf-8"

        def write(self, value: str) -> int:
            return len(value)

        def flush(self) -> None:
            return None

    artifact_path = tmp_path / "worker.log"
    with artifact_path.open("wb") as artifact:
        budget = LogBudget(8)
        stdout = BoundedTee(Sink(), artifact, budget)
        stderr = BoundedTee(Sink(), artifact, budget)
        stdout.write("123456")
        stderr.write("abcdef")
    data = artifact_path.read_bytes()
    assert data.startswith(b"123456ab")
    assert data.count(b"worker log truncated") == 1


def test_main_persists_bounded_log_and_exact_failure_window(
    monkeypatch: Any, tmp_path: Path
) -> None:
    evaluation = "eval-" + "2" * 20
    config = {
        "evaluation_id": evaluation,
        "request": {"target": "fixture", "model": "fixture"},
        "endpoint_identity": "http://fixture/v1/completions",
        "effective_configuration": {"profile": "gsm8k"},
    }
    monkeypatch.setenv("LM_EVAL_EVALUATION_CONFIG", json.dumps(config))
    monkeypatch.setenv("LM_EVAL_RESULT_ROOT", str(tmp_path))
    timestamps = iter(
        [
            datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 17, 12, 1, tzinfo=UTC),
        ]
    )
    clocks = iter([10.0, 25.0])
    monkeypatch.setattr(worker, "_timestamp", lambda: next(timestamps))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clocks))

    def fail(_config: dict[str, Any], _root: Path) -> None:
        print("durable diagnostic")
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(worker, "run", fail)
    with pytest.raises(SystemExit):
        worker.main()
    output = tmp_path / evaluation
    report = json.loads((output / "report.json").read_text())
    assert report["latency_seconds"] == 15.0
    assert report["measurement_started_at"] != report["measurement_completed_at"]
    log = (output / "worker.log").read_text()
    assert "durable diagnostic" in log
    assert "fixture failure" in log
    manifest = json.loads((output / "artifacts.json").read_text())
    assert "worker.log" in {item["name"] for item in manifest["artifacts"]}
    assert "worker.log" in report["artifact_hashes"]
