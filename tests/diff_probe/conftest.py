from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from nm_hard_tools.diff_probe.api import create_app
from nm_hard_tools.diff_probe.config import ProbeServiceSettings
from nm_hard_tools.diff_probe.kube import CANCELLED_AT_ANNOTATION

if TYPE_CHECKING:
    from pathlib import Path


class ApiError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status


class FakeKubernetes:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.creates = 0
        self.logs = "bounded worker log"

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        name = body["metadata"]["name"]
        if name in self.jobs:
            raise ApiError(409)
        self.creates += 1
        job = copy.deepcopy(body)
        job["metadata"]["creationTimestamp"] = datetime.now(UTC).isoformat()
        job["metadata"]["resourceVersion"] = "1"
        job["status"] = {}
        self.jobs[name] = job
        return copy.deepcopy(job)

    def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        if name not in self.jobs:
            raise ApiError(404)
        return copy.deepcopy(self.jobs[name])

    def list_jobs(
        self, namespace: str, label_selector: str, limit: int | None
    ) -> list[dict[str, Any]]:
        jobs = list(self.jobs.values())
        if limit is not None:
            jobs = jobs[:limit]
        return [copy.deepcopy(value) for value in jobs]

    def cancel_job(
        self, namespace: str, name: str, cancelled_at: str, resource_version: str
    ) -> dict[str, Any]:
        job = self.jobs[name]
        if job["metadata"]["resourceVersion"] != resource_version:
            raise ApiError(409)
        job["metadata"]["annotations"][CANCELLED_AT_ANNOTATION] = cancelled_at
        job["metadata"]["resourceVersion"] = str(int(resource_version) + 1)
        job["spec"]["suspend"] = True
        return copy.deepcopy(job)

    def wait_for_job_quiescence(
        self, namespace: str, probe_id: str, timeout_seconds: float
    ) -> dict[str, Any]:
        job = self.jobs[probe_id]
        job.setdefault("status", {}).pop("active", None)
        return copy.deepcopy(job)

    def read_logs(self, namespace: str, probe_id: str, tail_lines: int) -> str:
        return self.logs

    def ready(self, namespace: str) -> bool:
        return True


REQUEST_SET_ROWS = [
    {"id": "r1", "messages": [{"role": "user", "content": "Describe the image."}]},
    {"id": "r2", "messages": [{"role": "user", "content": "What tool applies?"}]},
]


def request_set_bytes() -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in REQUEST_SET_ROWS
    ).encode()


@pytest.fixture
def settings(tmp_path: Path) -> ProbeServiceSettings:
    result_root = tmp_path / "results"
    result_root.mkdir()
    request_sets = tmp_path / "request-sets"
    request_sets.mkdir()
    payload = request_set_bytes()
    (request_sets / "gemma4-vision.jsonl").write_bytes(payload)
    return ProbeServiceSettings.model_validate(
        {
            "namespace": "probes",
            "worker_image": "registry.example/diff-probe@sha256:" + "a" * 64,
            "result_claim_name": "probe-results",
            "result_root": str(result_root),
            "request_set_claim_name": "probe-request-sets",
            "request_set_root": str(request_sets),
            "worker_service_account": "probe-worker",
            "service_instance": "diff-probe-test",
            "image_pull_secrets": ["registry-pull"],
            "node_selector": {"kubernetes.io/arch": "amd64"},
            "targets": {
                "baseline-default": {
                    "base_url": "http://baseline-a.rig.svc:8000/v1",
                    "auth_secret_name": "probe-key",
                    "auth_secret_key": "token",
                },
                "baseline-reference": {
                    "base_url": "http://baseline-b.rig.svc:8000/v1",
                },
            },
            "request_sets": {
                "gemma4-vision": {
                    "file": "gemma4-vision.jsonl",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "api": "chat",
                    "max_requests": 64,
                    "default_max_tokens": 256,
                }
            },
        }
    )


@pytest.fixture
def kube() -> FakeKubernetes:
    return FakeKubernetes()


@pytest.fixture
def client(settings: ProbeServiceSettings, kube: FakeKubernetes) -> TestClient:
    return TestClient(
        create_app(settings, kube, "test-token"),
        headers={"Authorization": "Bearer test-token"},
    )


@pytest.fixture
def request_body() -> dict[str, Any]:
    return {
        "mode": "diff",
        "target_a": "baseline-default",
        "target_b": "baseline-reference",
        "model": "google/gemma-4-27b-it",
        "request_set": "gemma4-vision",
        "timeout_seconds": 3600,
    }
