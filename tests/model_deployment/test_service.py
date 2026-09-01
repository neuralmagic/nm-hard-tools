from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.models import ResourceRef
from nm_hard_tools.model_deployment.renderer import (
    OperatorConfigurationError,
    RenderedDeployment,
    WorkloadExpectation,
)
from nm_hard_tools.model_deployment.runtime import RuntimeFailure
from nm_hard_tools.model_deployment.service import DeployFailure, DeploymentService

FIXTURES = Path(__file__).parent / "fixtures"


class FakeRenderer:
    def render(self, _source: str) -> RenderedDeployment:
        return RenderedDeployment(
            manifesto_digest="a" * 64,
            intent_digest="c" * 64,
            deployment_id="hard-" + "b" * 24,
            namespace="models",
            endpoint="http://model.models.svc.cluster.local:8000/v1/models",
            objects=[
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "model"},
                }
            ],
            resources=[
                ResourceRef(api_version="apps/v1", kind="Deployment", name="model")
            ],
            workloads=[
                WorkloadExpectation(
                    name="model",
                    expected_pods=1,
                    pod_selector={"app.kubernetes.io/instance": "hard-" + "b" * 24},
                )
            ],
        )


class InvalidOperatorRenderer:
    def render(self, _source: str) -> RenderedDeployment:
        raise OperatorConfigurationError("invalid operator context")


class FakeRuntime:
    def __init__(self, failure: RuntimeFailure | None = None) -> None:
        self.failure = failure
        self.calls = 0
        self.active = 0
        self.maximum_active = 0

    def deploy(self, _rendered: RenderedDeployment, _timeout: int) -> None:
        self.calls += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.01)
        self.active -= 1
        if self.failure:
            raise self.failure


def settings(tmp_path: Path) -> DeploymentSettings:
    cluster = tmp_path / "cluster.yaml"
    cluster.write_text("name: test\n")
    token = tmp_path / "token"
    token.write_text("secret")
    return DeploymentSettings(
        cluster_profile=cluster,
        namespace="models",
        bearer_token_file=token,
        readiness_timeout_seconds=30,
    )


def test_success_returns_bounded_identity_and_serializes_identical_calls(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    service = DeploymentService(settings(tmp_path), runtime, FakeRenderer())
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.deploy, ["release: one", "release: one"]))
    assert {result.deployment_id for result in results} == {"hard-" + "b" * 24}
    assert all(result.ready for result in results)
    assert runtime.maximum_active == 1


@pytest.mark.parametrize(
    ("failure", "expected_id"),
    [
        (RuntimeFailure("APPLY_FAILED", "apply failed", False, False), None),
        (
            RuntimeFailure("DEPENDENCY_UNAVAILABLE", "unavailable", True, True),
            "hard-" + "b" * 24,
        ),
        (
            RuntimeFailure("READINESS_TIMEOUT", "not ready", True, False),
            "hard-" + "b" * 24,
        ),
    ],
)
def test_runtime_failures_preserve_post_acceptance_identity(
    tmp_path: Path, failure: RuntimeFailure, expected_id: str | None
) -> None:
    service = DeploymentService(
        settings(tmp_path), FakeRuntime(failure), FakeRenderer()
    )
    with pytest.raises(DeployFailure) as raised:
        service.deploy("release: one")
    assert raised.value.deployment_id == expected_id
    assert raised.value.retryable is failure.retryable


def test_invalid_operator_context_has_distinct_nonretryable_error(
    tmp_path: Path,
) -> None:
    service = DeploymentService(
        settings(tmp_path), FakeRuntime(), InvalidOperatorRenderer()
    )
    with pytest.raises(DeployFailure) as raised:
        service.deploy("release: one")
    assert raised.value.code == "INVALID_OPERATOR_CONFIGURATION"
    assert raised.value.retryable is False
    assert raised.value.deployment_id is None


def test_unschedulable_lws_name_is_rejected_without_touching_the_cluster(
    tmp_path: Path,
) -> None:
    """An over-long LeaderWorkerSet name must never reach the create boundary."""

    pytest.importorskip("manifesto")
    profile = tmp_path / "cluster.yaml"
    profile.write_text((FIXTURES / "cluster.yaml").read_text(), encoding="utf-8")
    token = tmp_path / "token"
    token.write_text("secret")
    runtime = FakeRuntime()
    service = DeploymentService(
        DeploymentSettings(
            cluster_profile=profile,
            namespace="models",
            bearer_token_file=token,
            readiness_timeout_seconds=30,
        ),
        runtime,
    )
    source = (
        (FIXTURES / "model.yaml")
        .read_text()
        .replace(
            "  - name: decode\n",
            "  - name: decode-with-a-very-long-role-name\n"
            "    workload: leaderworkerset\n",
        )
    )

    with pytest.raises(DeployFailure) as raised:
        service.deploy(source)

    assert raised.value.code == "INVALID_MANIFESTO_CONFIG"
    assert raised.value.retryable is False
    assert raised.value.deployment_id is None
    assert raised.value.field == "roles[0].name"
    assert runtime.calls == 0
