from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

HELM = shutil.which("helm")
CHART = Path(__file__).parents[2] / "charts" / "lm-eval-service"
REQUIRED_VALUES = [
    "--set=image.repository=registry.example/inference-hard-lm-eval-service",
    "--set=image.digest=sha256:" + "a" * 64,
    "--set=auth.existingSecret=lm-eval-api-token",
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


def test_config_changes_controller_pod_checksum() -> None:
    first = _render("gsm8k", "--set=auth.mutationRateLimitPerMinute=20")
    second = _render("gsm8k", "--set=auth.mutationRateLimitPerMinute=21")

    def checksum(documents: list[dict[str, Any]]) -> str:
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        return deployment["spec"]["template"]["metadata"]["annotations"][
            "checksum/config"
        ]

    assert checksum(first) != checksum(second)


def test_long_release_renders_valid_generated_names() -> None:
    documents = _render("r" * 53)
    assert all(len(item["metadata"]["name"]) <= 63 for item in documents)
    for kind in ("ServiceAccount", "NetworkPolicy"):
        names = [item["metadata"]["name"] for item in documents if item["kind"] == kind]
        assert len(names) == 2
        assert len(set(names)) == 2
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert len(settings["result_claim_name"]) <= 63
    assert len(settings["worker_service_account"]) <= 63


def test_container_build_is_locked_and_requires_explicit_base() -> None:
    dockerfile = (CHART.parents[1] / "Dockerfile.lm-eval-service").read_text()
    assert dockerfile.startswith("ARG BASE_IMAGE\n")
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "uv sync --frozen" in dockerfile


def test_release_selectors_are_isolated() -> None:
    for release in ("alpha", "beta"):
        documents = _render(release)
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        service = next(item for item in documents if item["kind"] == "Service")
        policies = [item for item in documents if item["kind"] == "NetworkPolicy"]

        assert (
            deployment["spec"]["selector"]["matchLabels"]["app.kubernetes.io/instance"]
            == release
        )
        assert service["spec"]["selector"]["app.kubernetes.io/instance"] == release
        assert all(
            policy["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/instance"]
            == release
            for policy in policies
        )


def test_private_registry_secret_reaches_controller_and_worker_config() -> None:
    documents = _render("gsm8k", "--set=imagePullSecrets[0].name=registry-pull")
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    assert deployment["spec"]["template"]["spec"]["imagePullSecrets"] == [
        {"name": "registry-pull"}
    ]
    accounts = [item for item in documents if item["kind"] == "ServiceAccount"]
    assert all("imagePullSecrets" not in account for account in accounts)
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert settings["image_pull_secrets"] == ["registry-pull"]


def test_node_selector_reaches_controller_and_worker_config() -> None:
    documents = _render("gsm8k")
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    assert deployment["spec"]["template"]["spec"]["nodeSelector"] == {
        "kubernetes.io/arch": "amd64"
    }
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert settings["node_selector"] == {"kubernetes.io/arch": "amd64"}


def test_chart_has_no_personal_image_or_target_defaults() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    assert values["image"]["repository"] == ""
    assert values["targets"] == {}
    assert values["profiles"]["gsm8k"]["tasks"][0]["dataset"]["revision"]


def test_shared_dataset_cache_is_explicit_and_read_only() -> None:
    documents = _render("evals", "--set=datasetCache.existingClaim=lm-eval-datasets")
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert settings["dataset_cache_claim_name"] == "lm-eval-datasets"
    assert settings["profiles"]["gsm8k"]["tasks"][0]["dataset"]["path"] == (
        "openai/gsm8k"
    )


def test_hf_token_secret_is_a_global_profile_fallback() -> None:
    documents = _render("evals", "--set=huggingFace.tokenSecret.name=hf-secret")
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    settings = yaml.safe_load(config_map["data"]["config.yaml"])
    assert settings["hf_token_secret_name"] == "hf-secret"
    assert settings["hf_token_secret_key"] == "HF_TOKEN"
