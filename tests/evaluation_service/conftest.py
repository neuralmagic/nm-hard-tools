from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from inference_hard_tools.evaluation_service.api import create_app
from inference_hard_tools.evaluation_service.config import ServiceSettings

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
        self.quiescence_hook: Any = None
        self.list_calls: list[tuple[str, str, int | None]] = []

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
        self.list_calls.append((namespace, label_selector, limit))
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
        job["metadata"]["annotations"]["evaluation.lm-eval.io/cancelled-at"] = (
            cancelled_at
        )
        job["metadata"]["resourceVersion"] = str(int(resource_version) + 1)
        job["spec"]["suspend"] = True
        return copy.deepcopy(job)

    def wait_for_job_quiescence(
        self, namespace: str, evaluation_id: str, timeout_seconds: float
    ) -> dict[str, Any]:
        if self.quiescence_hook is not None:
            self.quiescence_hook()
        job = self.jobs[evaluation_id]
        job.setdefault("status", {}).pop("active", None)
        return copy.deepcopy(job)

    def read_logs(self, namespace: str, evaluation_id: str, tail_lines: int) -> str:
        return self.logs

    def ready(self, namespace: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def baked_source_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EVAL_COMMIT", "8a07e1110d060de48cfc7a9a7987b7659060b60b")


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    result_root = tmp_path / "results"
    result_root.mkdir()
    return ServiceSettings.model_validate(
        {
            "namespace": "evals",
            "worker_image": "registry.example/lm-eval@sha256:" + "a" * 64,
            "result_claim_name": "eval-results",
            "result_root": str(result_root),
            "worker_service_account": "eval-worker",
            "service_instance": "lm-eval-test",
            "image_pull_secrets": ["registry-pull"],
            "node_selector": {"kubernetes.io/arch": "amd64"},
            "targets": {
                "kimi-k3": {
                    "base_url": "http://kimi-k3.default.svc:8000/v1",
                    "auth_secret_name": "kimi-key",
                    "auth_secret_key": "token",
                    "max_concurrency": 4,
                }
            },
            "profiles": {
                "gsm8k": {
                    "tasks": [
                        {
                            "name": "gsm8k",
                            "config_path": "lm_eval/tasks/gsm8k/gsm8k.yaml",
                            "dataset": {
                                "path": "openai/gsm8k",
                                "name": "main",
                                "revision": "740312add88f781978c0658806c59bc2815b9866",
                            },
                        }
                    ],
                    "primary_metric": {
                        "task": "gsm8k",
                        "metric": "exact_match",
                        "filter": "flexible-extract",
                    },
                    "default_num_fewshot": 5,
                    "max_samples": 1319,
                }
            },
        }
    )


@pytest.fixture
def kube() -> FakeKubernetes:
    return FakeKubernetes()


@pytest.fixture
def client(settings: ServiceSettings, kube: FakeKubernetes) -> TestClient:
    return TestClient(
        create_app(settings, kube, "test-token"),
        headers={"Authorization": "Bearer test-token"},
    )


@pytest.fixture
def request_body() -> dict[str, Any]:
    return {
        "target": "kimi-k3",
        "model": "mgoin/Kimi-K3-pruned75",
        "profile": "gsm8k",
        "num_fewshot": 5,
        "limit": 3,
        "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "generation_limit": 512,
        "chat_template": "none",
        "timeout_seconds": 3600,
        "workstream": {"workstream_id": "ws-0123456789ab", "run_id": "run-1"},
    }
