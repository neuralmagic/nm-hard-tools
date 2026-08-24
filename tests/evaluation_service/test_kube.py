from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from inference_hard_tools.evaluation_service.kube import KubernetesClient


class FakeApiClient:
    @staticmethod
    def sanitize_for_serialization(value: Any) -> dict[str, Any]:
        return value


class FakeBatch:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def list_namespaced_job(
        self,
        namespace: str,
        *,
        label_selector: str,
        limit: int,
        _continue: str | None,
    ) -> SimpleNamespace:
        self.calls.append((limit, _continue))
        if _continue is None:
            return SimpleNamespace(
                items=[{"metadata": {"name": "first"}}],
                metadata=SimpleNamespace(_continue="next-page"),
            )
        return SimpleNamespace(
            items=[{"metadata": {"name": "second"}}],
            metadata=SimpleNamespace(_continue=None),
        )


def test_list_jobs_consumes_all_kubernetes_pages() -> None:
    client = KubernetesClient.__new__(KubernetesClient)
    client._api_client = FakeApiClient()
    client._batch = FakeBatch()

    jobs = client.list_jobs("evals", "managed=true", None)

    assert [job["metadata"]["name"] for job in jobs] == ["first", "second"]
    assert client._batch.calls == [(500, None), (500, "next-page")]
