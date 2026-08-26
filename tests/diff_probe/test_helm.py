from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from nm_hard_tools.diff_probe.config import ProbeServiceSettings

HELM = shutil.which("helm")
CHART = Path(__file__).parents[2] / "charts" / "diff-probe-service"
REQUIRED_VALUES = [
    "--set=image.repository=registry.example/inference-hard-diff-probe-service",
    "--set=image.digest=sha256:" + "a" * 64,
    "--set=auth.existingSecret=diff-probe-api-token",
    "--set=requestSetStorage.existingClaim=probe-request-sets",
]


def _render(release: str, *values: str) -> list[dict[str, Any]]:
    if HELM is None:
        pytest.skip("helm is not installed")
    rendered = subprocess.run(
        [HELM, "template", release, str(CHART), *REQUIRED_VALUES, *values],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [document for document in yaml.safe_load_all(rendered) if document]


def test_rendered_config_is_a_valid_service_settings_document() -> None:
    documents = _render(
        "probes",
        "--set-json=targets={"
        '"baseline-default": {"base_url": "http://baseline-a.rig.svc:8000/v1"}}',
        "--set-json=requestSets={"
        '"gemma4-vision": {"file": "gemma4-vision.jsonl", "sha256": "'
        + "b" * 64
        + '"}}',
    )
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    parsed = ProbeServiceSettings.model_validate(settings)
    assert parsed.request_sets["gemma4-vision"].sha256 == "b" * 64
    assert parsed.targets["baseline-default"].base_url.startswith("http://")
    assert parsed.request_set_claim_name == "probe-request-sets"


def test_config_changes_controller_pod_checksum() -> None:
    first = _render("probes", "--set=auth.mutationRateLimitPerMinute=20")
    second = _render("probes", "--set=auth.mutationRateLimitPerMinute=21")

    def checksum(documents: list[dict[str, Any]]) -> str:
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        return deployment["spec"]["template"]["metadata"]["annotations"][
            "checksum/config"
        ]

    assert checksum(first) != checksum(second)


def test_long_release_renders_valid_generated_names() -> None:
    documents = _render("r" * 53)
    assert all(len(item["metadata"]["name"]) <= 63 for item in documents)
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert len(settings["result_claim_name"]) <= 63
    assert len(settings["worker_service_account"]) <= 63


def test_controller_env_names_the_probe_service() -> None:
    documents = _render("probes")
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"] for item in container["env"]}
    assert env == {"DIFF_PROBE_SERVICE_CONFIG", "DIFF_PROBE_API_TOKEN"}


def test_container_build_is_locked_and_requires_explicit_base() -> None:
    dockerfile = (CHART.parents[1] / "Dockerfile.diff-probe-service").read_text()
    assert dockerfile.startswith("ARG BASE_IMAGE\n")
