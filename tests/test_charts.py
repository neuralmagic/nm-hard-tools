from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")


def render(chart: str, *values: str) -> list[dict]:
    result = subprocess.run(
        ["helm", "template", chart, str(ROOT / "charts" / chart), *values],
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
