from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")


def render(chart: str, *values: str, release: str | None = None) -> list[dict]:
    result = subprocess.run(
        ["helm", "template", release or chart, str(ROOT / "charts" / chart), *values],
        check=True,
        capture_output=True,
        text=True,
    )
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def test_forward_pass_exposes_a_separate_mcp_sidecar() -> None:
    documents = render(
        "forward-pass-benchmark",
        "--set=image=example/forward:dev",
        "--set=mcpAdapter.image=example/adapter:dev",
        "--set=workerPlugin.image=example/plugin:dev",
        "--set=workerPlugin.bundleSha256=" + "a" * 64,
    )
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    containers = {
        item["name"]: item
        for item in deployment["spec"]["template"]["spec"]["containers"]
    }
    assert containers["mcp-adapter"]["command"] == [
        "inference-hard-forward-benchmark-mcp"
    ]
    environment = {
        item["name"]: item["value"] for item in containers["mcp-adapter"]["env"]
    }
    assert environment["INFERENCE_HARD_FORWARD_MCP_UPSTREAM"] == (
        "http://127.0.0.1:8000"
    )
    service = next(item for item in documents if item["kind"] == "Service")
    assert {item["name"]: item["port"] for item in service["spec"]["ports"]} == {
        "http": 8000,
        "mcp": 8001,
    }


@pytest.mark.parametrize(
    ("chart", "values"),
    [
        (
            "nyann-bench",
            (
                "--set=image=ghcr.io/neuralmagic/nyann-bench@sha256:" + "a" * 64,
                "--set=runnerImage=ghcr.io/neuralmagic/nyann-bench@sha256:" + "a" * 64,
                "--set=auth.existingSecret=token",
                "--set=storage.resultsClaim=results",
                "--set=storage.datasetClaim=datasets",
            ),
        ),
        (
            "agentx-service",
            (
                "--set=image=example/agentx@sha256:" + "a" * 64,
                "--set=runnerImage=example/runner@sha256:" + "b" * 64,
                "--set=auth.existingSecret=token",
                "--set=storage.resultsClaim=results",
            ),
        ),
    ],
)
def test_service_charts_require_immutable_images(
    chart: str, values: tuple[str, ...]
) -> None:
    render(chart, *values)
    mutable = [value.replace("@sha256:" + "a" * 64, ":latest") for value in values]
    result = subprocess.run(
        ["helm", "template", chart, str(ROOT / "charts" / chart), *mutable],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "digest pinned" in result.stderr


@pytest.mark.parametrize(
    ("chart", "values"),
    [
        (
            "forward-pass-benchmark",
            (
                "--set=image=example/forward:dev",
                "--set=mcpAdapter.image=example/adapter:dev",
                "--set=workerPlugin.image=example/plugin:dev",
                "--set=workerPlugin.bundleSha256=" + "a" * 64,
            ),
        ),
        (
            "lm-eval-service",
            (
                "--set=image.repository=example/lm-eval",
                "--set=image.digest=sha256:" + "a" * 64,
                "--set=auth.existingSecret=token",
            ),
        ),
    ],
)
def test_charts_support_operator_owned_rbac(
    chart: str, values: tuple[str, ...]
) -> None:
    documents = render(chart, *values, "--set=rbac.create=false")
    assert not any(item["kind"] in {"Role", "RoleBinding"} for item in documents)


MODEL_DEPLOYMENT_VALUES = (
    "--set=image.repository=example/model-deployment",
    "--set=image.digest=sha256:" + "a" * 64,
    "--set=auth.existingSecret=model-deployment-token",
    "--set=target.namespace=models",
    "--set-string=target.clusterProfile=name: test",
)


def test_model_deployment_chart_pins_image_and_keeps_target_operator_owned() -> None:
    documents = render("model-deployment", *MODEL_DEPLOYMENT_VALUES)
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "example/model-deployment@sha256:" + "a" * 64
    role = next(item for item in documents if item["kind"] == "Role")
    assert role["metadata"]["namespace"] == "models"
    settings = next(item for item in documents if item["kind"] == "ConfigMap")["data"][
        "settings.yaml"
    ]
    assert 'namespace: "models"' in settings
    assert "cluster_profile: /etc/nm-hard-tools/cluster-profile.yaml" in settings


def test_model_deployment_service_only_selects_its_own_release() -> None:
    """Two releases in one namespace must not cross-select each other's pods.

    A Service selecting on ``app.kubernetes.io/name`` alone fronts every release
    of the chart in the namespace, so a caller can reach a controller carrying a
    different operator-controlled ``target.clusterProfile``.
    """
    selectors = {}
    for release in ("rel-a", "rel-b"):
        documents = render(
            "model-deployment", *MODEL_DEPLOYMENT_VALUES, release=release
        )
        service = next(item for item in documents if item["kind"] == "Service")
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        selector = service["spec"]["selector"]
        assert selector["app.kubernetes.io/instance"] == release
        assert deployment["spec"]["selector"]["matchLabels"] == selector
        assert deployment["spec"]["template"]["metadata"]["labels"] == selector
        selectors[release] = selector

    assert selectors["rel-a"] != selectors["rel-b"]
