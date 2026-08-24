"""Small Kubernetes boundary, intentionally easy to fake in contract tests."""

from __future__ import annotations

import time
from typing import Any, Protocol


class KubernetesBackend(Protocol):
    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]: ...

    def get_job(self, namespace: str, name: str) -> dict[str, Any]: ...

    def list_jobs(
        self, namespace: str, label_selector: str, limit: int | None
    ) -> list[dict[str, Any]]: ...

    def cancel_job(
        self, namespace: str, name: str, cancelled_at: str, resource_version: str
    ) -> dict[str, Any]: ...

    def wait_for_job_quiescence(
        self, namespace: str, evaluation_id: str, timeout_seconds: float
    ) -> dict[str, Any]: ...

    def read_logs(self, namespace: str, evaluation_id: str, tail_lines: int) -> str: ...

    def ready(self, namespace: str) -> bool: ...


class KubernetesClient:
    """Production implementation using the official Kubernetes client."""

    def __init__(self) -> None:
        from kubernetes import client, config

        config.load_incluster_config()
        self._api_client = client.ApiClient()
        self._batch = client.BatchV1Api(self._api_client)
        self._core = client.CoreV1Api(self._api_client)

    def _dict(self, value: Any) -> dict[str, Any]:
        return self._api_client.sanitize_for_serialization(value)

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._dict(self._batch.create_namespaced_job(namespace, body))

    def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        return self._dict(self._batch.read_namespaced_job(name, namespace))

    def list_jobs(
        self, namespace: str, label_selector: str, limit: int | None
    ) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        continuation: str | None = None
        while True:
            page_limit = min(limit - len(jobs), 500) if limit is not None else 500
            result = self._batch.list_namespaced_job(
                namespace,
                label_selector=label_selector,
                limit=page_limit,
                _continue=continuation,
            )
            jobs.extend(self._dict(item) for item in result.items)
            continuation = getattr(result.metadata, "_continue", None)
            if not continuation or (limit is not None and len(jobs) >= limit):
                return jobs[:limit] if limit is not None else jobs

    def cancel_job(
        self, namespace: str, name: str, cancelled_at: str, resource_version: str
    ) -> dict[str, Any]:
        body = {
            "metadata": {
                "resourceVersion": resource_version,
                "annotations": {"evaluation.lm-eval.io/cancelled-at": cancelled_at},
            },
            "spec": {"suspend": True},
        }
        return self._dict(self._batch.patch_namespaced_job(name, namespace, body))

    def wait_for_job_quiescence(
        self, namespace: str, evaluation_id: str, timeout_seconds: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        selector = f"evaluation.lm-eval.io/evaluation-id={evaluation_id}"
        while True:
            job = self.get_job(namespace, evaluation_id)
            pods = self._core.list_namespaced_pod(
                namespace, label_selector=selector
            ).items
            live = any(
                str(pod.status.phase).lower() in {"pending", "running"} for pod in pods
            )
            if not live:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"worker for {evaluation_id} is still terminating")
            time.sleep(0.25)

    def read_logs(self, namespace: str, evaluation_id: str, tail_lines: int) -> str:
        pods = self._core.list_namespaced_pod(
            namespace,
            label_selector=f"evaluation.lm-eval.io/evaluation-id={evaluation_id}",
            limit=1,
        ).items
        if not pods:
            return ""
        return self._core.read_namespaced_pod_log(
            pods[0].metadata.name,
            namespace,
            tail_lines=tail_lines,
            timestamps=True,
            limit_bytes=65_536,
        )

    def ready(self, namespace: str) -> bool:
        self._batch.list_namespaced_job(namespace, limit=1)
        return True
