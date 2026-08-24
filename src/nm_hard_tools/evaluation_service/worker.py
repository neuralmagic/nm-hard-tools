"""Permissionless worker for one immutable operator-approved eval profile."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import math
import os
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lm_eval.utils import handle_non_serializable

from nm_hard_tools.evaluation_service import lm_eval_commit
from nm_hard_tools.evaluation_service.artifacts import artifact_metadata
from nm_hard_tools.evaluation_service.models import (
    EvaluationReport,
    EvaluationRequest,
    PrimaryMetric,
)

MAX_WORKER_LOG_BYTES = 64 * 1024 * 1024


class LogBudget:
    """One synchronized durable byte budget shared by stdout and stderr."""

    def __init__(self, remaining: int = MAX_WORKER_LOG_BYTES) -> None:
        self.remaining = remaining
        self.truncated = False
        self.lock = threading.Lock()


class BoundedTee:
    """Mirror text to Kubernetes logs and a bounded durable artifact."""

    def __init__(self, primary: Any, artifact: Any, budget: LogBudget) -> None:
        self.primary = primary
        self.artifact = artifact
        self.budget = budget

    @property
    def encoding(self) -> str:
        return self.primary.encoding or "utf-8"

    def write(self, value: str) -> int:
        written = self.primary.write(value)
        encoded = value.encode("utf-8", errors="replace")
        with self.budget.lock:
            available = self.budget.remaining
            if available:
                chunk = encoded[:available]
                self.artifact.write(chunk)
                self.budget.remaining -= len(chunk)
            if len(encoded) > available and not self.budget.truncated:
                self.artifact.write(b"\n[worker log truncated at 64 MiB]\n")
                self.budget.truncated = True
                self.budget.remaining = 0
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.artifact.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def fileno(self) -> int:
        return self.primary.fileno()


def _timestamp() -> datetime:
    return datetime.now(UTC)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=handle_non_serializable),
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_config(spec: dict[str, Any]) -> dict[str, Any]:
    """Load an installed task and bind it to its operator-pinned dataset."""
    from lm_eval.tasks._yaml_loader import load_yaml

    relative = Path(spec["config_path"]).relative_to("lm_eval")
    path = importlib.resources.files("lm_eval").joinpath(*relative.parts)
    task = load_yaml(path, resolve_func=True)
    if task.get("task") != spec["name"]:
        raise RuntimeError(
            f"installed task name {task.get('task')!r} does not match "
            f"profile task {spec['name']!r}"
        )
    dataset = spec["dataset"]
    task["dataset_path"] = dataset["path"]
    if dataset.get("name") is None:
        task.pop("dataset_name", None)
    else:
        task["dataset_name"] = dataset["name"]
    task["dataset_kwargs"] = {
        **(task.get("dataset_kwargs") or {}),
        "revision": dataset["revision"],
    }
    return task


def _load_tasks(
    specs: list[dict[str, Any]],
) -> tuple[Any, list[Any], dict[str, dict[str, str]]]:
    """Load each task once and retain its actual dataset split fingerprints."""
    from datasets import DatasetDict
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    loaded = manager.load([_task_config(spec) for spec in specs])
    tasks = loaded["tasks"]
    expected = {spec["name"] for spec in specs}
    if set(tasks) != expected:
        raise RuntimeError("loaded lm-eval task set does not match the profile")
    fingerprints: dict[str, dict[str, str]] = {}
    specs_by_name = {spec["name"]: spec for spec in specs}
    for task_name, task in tasks.items():
        dataset = specs_by_name[task_name]["dataset"]
        identity = (
            f"{dataset['path']}:{dataset.get('name') or '-'}@{dataset['revision']}"
        )
        loaded_dataset = task.dataset
        splits = (
            loaded_dataset.items()
            if isinstance(loaded_dataset, DatasetDict)
            else [("default", loaded_dataset)]
        )
        split_fingerprints: dict[str, str] = {}
        for split, value in splits:
            fingerprint = getattr(value, "_fingerprint", None)
            if not isinstance(fingerprint, str) or not fingerprint:
                raise RuntimeError(
                    f"dataset for task {task_name!r} has no runtime fingerprint"
                )
            split_fingerprints[str(split)] = fingerprint
        fingerprints[identity] = split_fingerprints
    return manager, list(tasks.values()), fingerprints


def summarize_metrics(
    result: dict[str, Any], primary: dict[str, str]
) -> tuple[dict[str, dict[str, float]], PrimaryMetric, dict[str, int]]:
    """Keep finite numeric lm-eval metrics and resolve the profile's primary one."""
    metrics: dict[str, dict[str, float]] = {}
    for task, values in result.get("results", {}).items():
        task_metrics: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                task_metrics[str(name)] = numeric
        metrics[str(task)] = task_metrics
    key = f"{primary['metric']},{primary['filter']}"
    try:
        primary_value = metrics[primary["task"]][key]
    except KeyError as exc:
        raise RuntimeError(
            f"primary metric {primary['task']}/{key} was not produced"
        ) from exc
    counts = {
        str(task): int(values.get("effective", 0))
        for task, values in result.get("n-samples", {}).items()
    }
    return (
        metrics,
        PrimaryMetric(**primary, value=primary_value),
        counts,
    )


def run(config: dict[str, Any], result_root: Path) -> EvaluationReport:
    """Execute the typed request via lm-eval's public evaluator API."""
    from lm_eval import simple_evaluate

    request = EvaluationRequest.model_validate(config["request"])
    evaluation = config["evaluation_id"]
    output_dir = result_root / evaluation
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    config_path = output_dir / "effective-configuration.json"
    _atomic_json(config_path, config["effective_configuration"])

    invocation = config["invocation"]
    task_manager, tasks, dataset_fingerprints = _load_tasks(invocation["tasks"])
    seeds = invocation["seeds"]
    started_at = _timestamp()
    clock_started = time.monotonic()
    result = simple_evaluate(
        model=invocation["model_backend"],
        model_args=invocation["model_args"],
        tasks=tasks,
        num_fewshot=invocation["num_fewshot"],
        batch_size=invocation["batch_size"],
        limit=invocation["limit"],
        bootstrap_iters=invocation["bootstrap_iters"],
        log_samples=invocation["log_samples"],
        apply_chat_template=invocation["apply_chat_template"],
        fewshot_as_multiturn=invocation["fewshot_as_multiturn"],
        gen_kwargs=invocation["generation_kwargs"],
        random_seed=seeds["python"],
        numpy_random_seed=seeds["numpy"],
        torch_random_seed=seeds["torch"],
        fewshot_random_seed=seeds["fewshot"],
        confirm_run_unsafe_code=invocation["confirm_run_unsafe_code"],
        cache_requests=invocation["cache_requests"],
        rewrite_requests_cache=invocation["rewrite_requests_cache"],
        delete_requests_cache=invocation["delete_requests_cache"],
        check_integrity=invocation["check_integrity"],
        write_out=invocation["write_out"],
        predict_only=invocation["predict_only"],
        system_instruction=invocation["system_instruction"],
        metadata=invocation["metadata"],
        task_manager=task_manager,
    )
    completed_at = _timestamp()
    latency = time.monotonic() - clock_started
    if result is None:
        raise RuntimeError("lm-eval returned no rank-zero result")

    metrics, primary_metric, sample_counts = summarize_metrics(
        result, invocation["primary_metric"]
    )
    samples = result.pop("samples", {})
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for task, task_samples in samples.items():
            for sample in task_samples:
                stream.write(
                    json.dumps(
                        {"task": task, **sample}, default=handle_non_serializable
                    )
                )
                stream.write("\n")
    raw_path = output_dir / "lm-eval-result.json"
    _atomic_json(raw_path, result)
    artifacts = [
        artifact_metadata(config_path, "application/json"),
        artifact_metadata(raw_path, "application/json"),
        artifact_metadata(samples_path, "application/x-ndjson"),
    ]
    baked_commit = lm_eval_commit()
    discovered_commit = result.get("git_hash")
    if (
        isinstance(discovered_commit, str)
        and len(discovered_commit) == 40
        and discovered_commit != baked_commit
    ):
        raise RuntimeError(
            "lm-eval runtime Git revision does not match the baked dependency revision"
        )
    report = EvaluationReport(
        evaluation_id=evaluation,
        terminal_state="succeeded",
        metrics=metrics,
        primary_metric=primary_metric,
        sample_counts=sample_counts,
        dataset_fingerprints=dataset_fingerprints,
        latency_seconds=latency,
        lm_eval_version=result.get(
            "lm_eval_version", importlib.metadata.version("lm_eval")
        ),
        lm_eval_commit=baked_commit,
        effective_configuration=config["effective_configuration"],
        model_identity={
            "served_model": request.model,
            "endpoint": config["endpoint_identity"],
        },
        measurement_started_at=started_at,
        measurement_completed_at=completed_at,
        artifact_hashes={item["name"]: item["sha256"] for item in artifacts},
    )
    report_path = output_dir / "report.json"
    _atomic_json(report_path, report.model_dump(mode="json"))
    artifacts.append(artifact_metadata(report_path, "application/json"))
    _atomic_json(
        output_dir / "artifacts.json",
        {"evaluation_id": evaluation, "artifacts": artifacts},
    )
    return report


def _write_failure(
    config: dict[str, Any],
    result_root: Path,
    exc: Exception,
    started_at: datetime,
    latency_seconds: float,
) -> None:
    evaluation = config["evaluation_id"]
    output_dir = result_root / evaluation
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    completed_at = _timestamp()
    error_path = output_dir / "error.txt"
    error_path.write_text(
        f"{type(exc).__name__}: {str(exc)[:4000]}\n", encoding="utf-8"
    )
    error_artifact = artifact_metadata(error_path, "text/plain")
    report = EvaluationReport(
        evaluation_id=evaluation,
        terminal_state="failed",
        metrics={},
        primary_metric=None,
        sample_counts={},
        dataset_fingerprints={},
        latency_seconds=latency_seconds,
        lm_eval_version=importlib.metadata.version("lm_eval"),
        lm_eval_commit=os.environ.get("LM_EVAL_COMMIT", "unknown"),
        effective_configuration=config.get("effective_configuration", {}),
        model_identity={
            "served_model": config.get("request", {}).get("model", "unknown"),
            "endpoint": config.get("endpoint_identity", "unknown"),
        },
        measurement_started_at=started_at,
        measurement_completed_at=completed_at,
        artifact_hashes={error_artifact["name"]: error_artifact["sha256"]},
        error=f"{type(exc).__name__}: {str(exc)[:4000]}",
    )
    report_path = output_dir / "report.json"
    _atomic_json(report_path, report.model_dump(mode="json"))
    artifacts = [
        error_artifact,
        artifact_metadata(report_path, "application/json"),
    ]
    _atomic_json(
        output_dir / "artifacts.json",
        {"evaluation_id": evaluation, "artifacts": artifacts},
    )


def _finalize_log_artifact(output_dir: Path) -> None:
    """Attach the closed durable worker log to report and artifact metadata."""
    log_path = output_dir / "worker.log"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "artifacts.json"
    if not log_path.exists() or not report_path.exists() or not manifest_path.exists():
        return
    log_artifact = artifact_metadata(log_path, "text/plain")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("artifact_hashes", {})[log_path.name] = log_artifact["sha256"]
    _atomic_json(report_path, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("name") not in {log_path.name, report_path.name}
    ]
    artifacts.extend([log_artifact, artifact_metadata(report_path, "application/json")])
    _atomic_json(
        manifest_path,
        {"evaluation_id": report["evaluation_id"], "artifacts": artifacts},
    )


def main() -> None:
    process_started_at = _timestamp()
    process_clock_started = time.monotonic()
    failure: Exception | None = None
    try:
        config = json.loads(os.environ["LM_EVAL_EVALUATION_CONFIG"])
        result_root = Path(
            os.environ.get("LM_EVAL_RESULT_ROOT", "/var/lib/lm-eval-results")
        )
        output_dir = result_root / config["evaluation_id"]
        output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
        with (output_dir / "worker.log").open("ab", buffering=0) as log_stream:
            budget = LogBudget()
            stdout = BoundedTee(sys.stdout, log_stream, budget)
            stderr = BoundedTee(sys.stderr, log_stream, budget)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    run(config, result_root)
                except Exception as exc:  # noqa: BLE001 - persist all worker failures.
                    failure = exc
                    traceback.print_exc()
                    _write_failure(
                        config,
                        result_root,
                        exc,
                        process_started_at,
                        time.monotonic() - process_clock_started,
                    )
                    print(
                        f"evaluation failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
    except Exception as exc:  # noqa: BLE001 - last-resort process boundary.
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        failure = exc
    finally:
        if "output_dir" in locals():
            _finalize_log_artifact(output_dir)
    if failure is not None:
        raise SystemExit(1) from failure


if __name__ == "__main__":
    main()
