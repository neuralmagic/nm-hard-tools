"""Permissionless worker for one immutable probe plan.

Verifies the frozen request set by content hash, runs it greedily against the
planned endpoint(s), computes bounded divergence/degeneracy metrics, and
persists content-hashed artifacts plus a durable terminal report. Any transport
failure fails the probe: a partial read is not a measurement.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from nm_hard_tools.diff_probe import SERVICE_VERSION
from nm_hard_tools.diff_probe.models import ProbeReport, ProbeRequest
from nm_hard_tools.evaluation_service.artifacts import artifact_metadata, atomic_json

MAX_RESPONSE_BYTES = 1_048_576
REQUEST_RETRIES = 2


class ProbeFailure(RuntimeError):
    """A condition that invalidates the whole measurement."""


# --- pure measurement logic ---------------------------------------------------


def repeated_tail(
    text: str, min_unit: int = 2, max_unit: int = 64, min_repeats: int = 4
) -> bool:
    """Whether the text ends in one unit repeated consecutively >= min_repeats
    times — the classic degenerate-generation tail."""
    tail = text[-(max_unit * min_repeats) :]
    for unit_len in range(min_unit, max_unit + 1):
        if len(tail) < unit_len * min_repeats:
            break
        unit = tail[-unit_len:]
        if unit.strip() and tail.endswith(unit * min_repeats):
            return True
    return False


def is_degenerate(text: str) -> bool:
    """Empty output, a repeated tail, or a near-single-symbol output."""
    stripped = text.strip()
    if not stripped:
        return True
    if repeated_tail(stripped):
        return True
    window = stripped[-256:]
    return len(window) >= 64 and len(set(window)) <= 4


def divergence(a: str, b: str) -> float:
    """1 - similarity ratio; 0.0 = identical, 1.0 = nothing in common."""
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def summarize_diff(pairs: list[tuple[str, str]]) -> dict[str, float]:
    n = len(pairs)
    if n == 0:
        raise ProbeFailure("request set produced no measurable pairs")
    divergences = [divergence(a, b) for a, b in pairs]
    return {
        "request_count": float(n),
        "mean_divergence": sum(divergences) / n,
        "max_divergence": max(divergences),
        "exact_match_rate": sum(1 for a, b in pairs if a == b) / n,
        "degenerate_rate_a": sum(1 for a, _ in pairs if is_degenerate(a)) / n,
        "degenerate_rate_b": sum(1 for _, b in pairs if is_degenerate(b)) / n,
    }


def summarize_smoke(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        raise ProbeFailure("request set produced no measurable responses")
    passed = sum(1 for row in rows if row["checks_passed"])
    return {
        "request_count": float(n),
        "pass_rate": passed / n,
        "degenerate_rate": sum(1 for r in rows if r["degenerate"]) / n,
        "empty_rate": sum(1 for r in rows if not r["text"].strip()) / n,
    }


def smoke_checks(text: str, finish_reason: str | None) -> tuple[bool, bool]:
    degenerate = is_degenerate(text)
    passed = bool(text.strip()) and finish_reason is not None and not degenerate
    return passed, degenerate


# --- request set --------------------------------------------------------------


def load_request_set(
    path: Path, expected_sha256: str, api: str, limit: int | None, max_requests: int
) -> list[dict[str, Any]]:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ProbeFailure(
            f"request set {path.name} content hash {actual[:12]} does not match "
            f"the pinned {expected_sha256[:12]}; refusing to measure a moved set"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ProbeFailure(f"request set line {index} is not an object")
        if api == "chat" and not isinstance(row.get("messages"), list):
            raise ProbeFailure(f"request set line {index} has no messages list")
        if api == "completions" and not isinstance(row.get("prompt"), str):
            raise ProbeFailure(f"request set line {index} has no prompt string")
        rows.append(row)
    if not rows:
        raise ProbeFailure("request set is empty")
    if len(rows) > max_requests:
        raise ProbeFailure(
            f"request set holds {len(rows)} rows, over the configured "
            f"maximum {max_requests}"
        )
    return rows[:limit] if limit is not None else rows


# --- transport ----------------------------------------------------------------


class Endpoint:
    def __init__(self, spec: dict[str, Any], timeout_seconds: float) -> None:
        self.spec = spec
        self.client = httpx.Client(
            timeout=httpx.Timeout(min(timeout_seconds, 300.0)),
            verify=spec.get("tls_verify", True),
        )

    def close(self) -> None:
        self.client.close()

    def generate(
        self, api: str, model: str, row: dict[str, Any], max_tokens: int
    ) -> tuple[str, str | None]:
        body: dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(row.get("max_tokens", max_tokens)),
        }
        if api == "chat":
            body["messages"] = row["messages"]
        else:
            body["prompt"] = row["prompt"]
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.spec.get("api_key_env", ""), "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        last_error: Exception | None = None
        for _attempt in range(REQUEST_RETRIES + 1):
            try:
                response = self.client.post(
                    self.spec["url"], json=body, headers=headers
                )
                response.raise_for_status()
                if len(response.content) > MAX_RESPONSE_BYTES:
                    raise ProbeFailure(
                        f"{self.spec['identity']} returned an oversized response"
                    )
                payload = response.json()
                choice = (payload.get("choices") or [{}])[0]
                if api == "chat":
                    text = (choice.get("message") or {}).get("content") or ""
                else:
                    text = choice.get("text") or ""
                return str(text), choice.get("finish_reason")
            except ProbeFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then terminal.
                last_error = exc
        raise ProbeFailure(
            f"request to {self.spec['identity']} failed after "
            f"{REQUEST_RETRIES + 1} attempts: {last_error}"
        ) from last_error


# --- run ----------------------------------------------------------------------


def _timestamp() -> datetime:
    return datetime.now(UTC)


def run(
    config: dict[str, Any], result_root: Path, request_set_root: Path
) -> ProbeReport:
    request = ProbeRequest.model_validate(config["request"])
    probe = config["probe_id"]
    output_dir = result_root / probe
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    config_path = output_dir / "effective-configuration.json"
    atomic_json(config_path, config["effective_configuration"])

    request_set = config["request_set"]
    rows = load_request_set(
        request_set_root / request_set["file"],
        request_set["sha256"],
        request_set["api"],
        request.limit,
        request_set["max_requests"],
    )
    api = request_set["api"]
    max_tokens = int(config["max_tokens"])

    endpoints = {
        role: Endpoint(spec, float(request.timeout_seconds))
        for role, spec in config["endpoints"].items()
    }
    samples_path = output_dir / "samples.jsonl"
    started_at = _timestamp()
    clock_started = time.monotonic()
    try:
        with samples_path.open("w", encoding="utf-8") as stream:
            if request.mode == "diff":
                pairs: list[tuple[str, str]] = []
                for index, row in enumerate(rows):
                    text_a, finish_a = endpoints["a"].generate(
                        api, request.model, row, max_tokens
                    )
                    text_b, finish_b = endpoints["b"].generate(
                        api, request.model, row, max_tokens
                    )
                    pairs.append((text_a, text_b))
                    stream.write(
                        json.dumps(
                            {
                                "index": index,
                                "id": row.get("id"),
                                "text_a": text_a,
                                "text_b": text_b,
                                "finish_reason_a": finish_a,
                                "finish_reason_b": finish_b,
                                "divergence": divergence(text_a, text_b),
                                "degenerate_a": is_degenerate(text_a),
                                "degenerate_b": is_degenerate(text_b),
                            },
                            sort_keys=True,
                        )
                    )
                    stream.write("\n")
                metrics = summarize_diff(pairs)
                primary = "mean_divergence"
            else:
                smoke_rows: list[dict[str, Any]] = []
                for index, row in enumerate(rows):
                    text, finish = endpoints["a"].generate(
                        api, request.model, row, max_tokens
                    )
                    passed, degenerate = smoke_checks(text, finish)
                    smoke_rows.append(
                        {
                            "text": text,
                            "checks_passed": passed,
                            "degenerate": degenerate,
                        }
                    )
                    stream.write(
                        json.dumps(
                            {
                                "index": index,
                                "id": row.get("id"),
                                "text": text,
                                "finish_reason": finish,
                                "checks_passed": passed,
                                "degenerate": degenerate,
                            },
                            sort_keys=True,
                        )
                    )
                    stream.write("\n")
                metrics = summarize_smoke(smoke_rows)
                primary = "pass_rate"
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
    completed_at = _timestamp()
    latency = time.monotonic() - clock_started

    metrics = {
        name: float(value)
        for name, value in metrics.items()
        if math.isfinite(float(value))
    }
    artifacts = [
        artifact_metadata(config_path, "application/json"),
        artifact_metadata(samples_path, "application/x-ndjson"),
    ]
    report = ProbeReport(
        probe_id=probe,
        terminal_state="succeeded",
        mode=request.mode,
        metrics=metrics,
        primary_metric=primary,
        request_count=len(rows),
        request_set_fingerprint={
            "name": request.request_set,
            "file": request_set["file"],
            "sha256": request_set["sha256"],
        },
        endpoint_identities={
            role: spec["identity"] for role, spec in config["endpoints"].items()
        },
        model_identity={"served_model": request.model},
        effective_configuration=config["effective_configuration"],
        measurement_started_at=started_at,
        measurement_completed_at=completed_at,
        latency_seconds=latency,
        artifact_hashes={item["name"]: item["sha256"] for item in artifacts},
        service_version=SERVICE_VERSION,
    )
    report_path = output_dir / "report.json"
    atomic_json(report_path, report.model_dump(mode="json"))
    artifacts.append(artifact_metadata(report_path, "application/json"))
    atomic_json(
        output_dir / "artifacts.json",
        {"probe_id": probe, "artifacts": artifacts},
    )
    return report


def _write_failure(
    config: dict[str, Any],
    result_root: Path,
    exc: Exception,
    started_at: datetime,
    latency_seconds: float,
) -> None:
    request = ProbeRequest.model_validate(config["request"])
    probe = config["probe_id"]
    output_dir = result_root / probe
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    error_path = output_dir / "error.txt"
    error_path.write_text(
        f"{type(exc).__name__}: {str(exc)[:4000]}\n", encoding="utf-8"
    )
    artifacts = [artifact_metadata(error_path, "text/plain")]
    report = ProbeReport(
        probe_id=probe,
        terminal_state="failed",
        mode=request.mode,
        metrics={},
        primary_metric="mean_divergence" if request.mode == "diff" else "pass_rate",
        request_count=0,
        request_set_fingerprint={
            "name": request.request_set,
            "file": config["request_set"]["file"],
            "sha256": config["request_set"]["sha256"],
        },
        endpoint_identities={
            role: spec["identity"] for role, spec in config["endpoints"].items()
        },
        model_identity={"served_model": request.model},
        effective_configuration=config["effective_configuration"],
        measurement_started_at=started_at,
        measurement_completed_at=_timestamp(),
        latency_seconds=latency_seconds,
        artifact_hashes={item["name"]: item["sha256"] for item in artifacts},
        service_version=SERVICE_VERSION,
        error=f"{type(exc).__name__}: {str(exc)[:4000]}",
    )
    atomic_json(output_dir / "report.json", report.model_dump(mode="json"))
    artifacts.append(artifact_metadata(output_dir / "report.json"))
    atomic_json(
        output_dir / "artifacts.json",
        {"probe_id": probe, "artifacts": artifacts},
    )


def main() -> int:
    config = json.loads(os.environ["DIFF_PROBE_CONFIG"])
    result_root = Path(os.environ["DIFF_PROBE_RESULT_ROOT"])
    request_set_root = Path(os.environ["DIFF_PROBE_REQUEST_SET_ROOT"])
    started = _timestamp()
    clock = time.monotonic()
    try:
        report = run(config, result_root, request_set_root)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a report.
        traceback.print_exc()
        _write_failure(config, result_root, exc, started, time.monotonic() - clock)
        return 1
    print(
        json.dumps(
            {
                "probe_id": report.probe_id,
                "terminal_state": report.terminal_state,
                "metrics": report.metrics,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
