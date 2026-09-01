from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.models import ResourceRef
from nm_hard_tools.model_deployment.renderer import (
    DIGEST_ANNOTATION,
    OWNER_ANNOTATION,
    RenderedDeployment,
    WorkloadExpectation,
)
from nm_hard_tools.model_deployment.runtime import (
    KubectlRuntime,
    RuntimeFailure,
    _pod_ready,
)


class FakeKubectlRuntime(KubectlRuntime):
    def __init__(self, settings: DeploymentSettings) -> None:
        super().__init__(settings)
        self.objects: dict[str, dict] = {}
        self.commands: list[list[str]] = []
        self.pods_by_selector: dict[str, list[dict]] = {}
        self.race_collision_name: str | None = None

    def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[:2] == ["get", "pods"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"items": self.pods_by_selector.get(args[3], [])}),
                "",
            )
        if args[0] == "get":
            body = yaml.safe_load(input_text or "{}")
            name = body["metadata"]["name"]
            value = self.objects.get(name)
            return subprocess.CompletedProcess(
                args, 0, json.dumps(value) if value is not None else "", ""
            )
        if args[:2] == ["create", "--dry-run=server"]:
            return subprocess.CompletedProcess(args, 0, "validated", "")
        if args[0] == "create":
            body = yaml.safe_load(input_text or "{}")
            name = body["metadata"]["name"]
            if name == self.race_collision_name:
                self.objects[name] = {
                    **body,
                    "metadata": {"name": name, "annotations": {}},
                }
                return subprocess.CompletedProcess(args, 1, "", "already exists")
            if name in self.objects:
                return subprocess.CompletedProcess(args, 1, "", "already exists")
            self.objects[name] = body
            return subprocess.CompletedProcess(args, 0, "created", "")
        raise AssertionError(args)

    @staticmethod
    def _endpoint_ready(_endpoint: str, _timeout_seconds: float) -> bool:
        return True


def settings(tmp_path: Path) -> DeploymentSettings:
    cluster = tmp_path / "cluster.yaml"
    cluster.write_text("name: test\n")
    token = tmp_path / "token"
    token.write_text("secret")
    return DeploymentSettings(
        cluster_profile=cluster.resolve(),
        namespace="models",
        bearer_token_file=token.resolve(),
        readiness_timeout_seconds=30,
    )


def rendered() -> RenderedDeployment:
    deployment_id = "hard-" + "b" * 24
    obj = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "model",
            "namespace": "models",
            "annotations": {
                OWNER_ANNOTATION: deployment_id,
                DIGEST_ANNOTATION: "a" * 64,
            },
        },
    }
    return RenderedDeployment(
        manifesto_digest="c" * 64,
        intent_digest="e" * 64,
        deployment_id=deployment_id,
        namespace="models",
        endpoint="http://model.models.svc.cluster.local:8000/v1/models",
        objects=[obj],
        resources=[ResourceRef(api_version="apps/v1", kind="Deployment", name="model")],
        workloads=[
            WorkloadExpectation(
                name="model",
                expected_pods=1,
                pod_selector={"app.kubernetes.io/instance": deployment_id},
            )
        ],
    )


def ready_pod(deployment_id: str, manifesto_digest: str) -> dict:
    return {
        "metadata": {
            "annotations": {
                OWNER_ANNOTATION: deployment_id,
                DIGEST_ANNOTATION: manifesto_digest,
            }
        },
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }


def test_kubectl_runtime_dry_runs_creates_once_and_observes_retry(
    tmp_path: Path,
) -> None:
    runtime = FakeKubectlRuntime(settings(tmp_path))
    value = rendered()
    runtime.pods_by_selector[f"app.kubernetes.io/instance={value.deployment_id}"] = [
        ready_pod(value.deployment_id, value.intent_digest)
    ]
    runtime.deploy(value, 30)
    runtime.deploy(value, 30)
    creates = [args for args in runtime.commands if args[0] == "create"]
    assert creates == [
        ["create", "--dry-run=server", "-f", "-"],
        ["create", "-f", "-"],
    ]


def test_kubectl_runtime_refuses_to_adopt_existing_resource(tmp_path: Path) -> None:
    runtime = FakeKubectlRuntime(settings(tmp_path))
    value = rendered()
    runtime.objects["model"] = {
        **value.objects[0],
        "metadata": {"name": "model", "annotations": {}},
    }
    with pytest.raises(RuntimeFailure) as raised:
        runtime.deploy(value, 30)
    assert raised.value.code == "APPLY_FAILED"
    assert raised.value.retryable is False
    assert not any(args[0] == "create" for args in runtime.commands)


def test_post_acceptance_collision_preserves_accepted_state(tmp_path: Path) -> None:
    runtime = FakeKubectlRuntime(settings(tmp_path))
    value = rendered()
    second = {
        **value.objects[0],
        "metadata": {
            **value.objects[0]["metadata"],
            "name": "second",
            "annotations": {
                OWNER_ANNOTATION: value.deployment_id,
                DIGEST_ANNOTATION: "d" * 64,
            },
        },
    }
    value = RenderedDeployment(
        **{**value.__dict__, "objects": [value.objects[0], second]}
    )
    runtime.race_collision_name = "second"
    with pytest.raises(RuntimeFailure) as raised:
        runtime.deploy(value, 30)
    assert raised.value.code == "APPLY_FAILED"
    assert raised.value.accepted is True


def test_each_rendered_workload_requires_its_own_ready_owned_pods(
    tmp_path: Path,
) -> None:
    runtime = FakeKubectlRuntime(settings(tmp_path))
    value = rendered()
    value = RenderedDeployment(
        **{
            **value.__dict__,
            "workloads": [
                value.workloads[0],
                WorkloadExpectation(
                    name="second",
                    expected_pods=1,
                    pod_selector={"llm-d.ai/role": "second"},
                ),
            ],
        }
    )
    runtime.pods_by_selector[f"app.kubernetes.io/instance={value.deployment_id}"] = [
        ready_pod(value.deployment_id, value.intent_digest)
    ]
    assert runtime._workload_ready(
        value.workloads[0].pod_selector,
        1,
        value.deployment_id,
        value.intent_digest,
        10**18,
    )
    assert not runtime._workload_ready(
        value.workloads[1].pod_selector,
        1,
        value.deployment_id,
        value.intent_digest,
        10**18,
    )


def test_pod_readiness_requires_matching_owner_and_manifesto_digest() -> None:
    deployment_id = "hard-" + "b" * 24
    manifesto_digest = "c" * 64
    pod = ready_pod(deployment_id, manifesto_digest)
    assert _pod_ready(pod, deployment_id, manifesto_digest)
    assert not _pod_ready(pod, deployment_id, "d" * 64)
    assert not _pod_ready({"status": pod["status"]}, deployment_id, manifesto_digest)
