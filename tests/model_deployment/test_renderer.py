from __future__ import annotations

from pathlib import Path

import pytest

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.renderer import (
    DIGEST_ANNOTATION,
    OWNER_ANNOTATION,
    ManifestoConfigError,
    ManifestoRenderer,
    parse_manifesto_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_safe_manifesto_yaml_accepts_one_self_contained_document() -> None:
    value = parse_manifesto_config(
        "release: qwen\ntopology: aggregated\nmodel:\n  id: org/model\n"
    )
    assert value["model"]["id"] == "org/model"


@pytest.mark.parametrize(
    "source",
    [
        "release: one\nrelease: two\n",
        "---\nrelease: one\n---\nrelease: two\n",
        "release: &name one\nmodel: *name\n",
        "release: !custom one\n",
        "? [unhashable, key]\n: value\n",
        "namespace: other\nrelease: one\n",
        "accelerator: b200\nrelease: one\n",
        "routing: {kind: disabled}\nrelease: one\n",
        "model:\n  hf_home: /cache\nrelease: one\n",
        "roles:\n  - fabric_profile: rdma\nrelease: one\n",
        "runtime:\n  env: {HF_TOKEN: secret}\nrelease: one\n",
        "release: one\nextends: base.yaml\n",
        "- not\n- an\n- object\n",
    ],
)
def test_safe_manifesto_yaml_rejects_ambiguous_or_operator_fields(source: str) -> None:
    with pytest.raises(ManifestoConfigError):
        parse_manifesto_config(source)


def test_safe_manifesto_yaml_bounds_expanded_nodes() -> None:
    source = "items:\n" + "".join(f"  - {index}\n" for index in range(10_001))
    with pytest.raises(ManifestoConfigError, match="10000-node"):
        parse_manifesto_config(source)


def test_safe_manifesto_yaml_rejects_pathological_nesting() -> None:
    source = "value: " + "[" * 2_000 + "0" + "]" * 2_000
    with pytest.raises(ManifestoConfigError):
        parse_manifesto_config(source)


def test_pinned_manifesto_renders_one_owned_direct_endpoint(tmp_path: Path) -> None:
    pytest.importorskip("manifesto")
    token = tmp_path / "token"
    token.write_text("secret")
    settings = DeploymentSettings(
        cluster_profile=(FIXTURES / "cluster.yaml").resolve(),
        namespace="models",
        bearer_token_file=token.resolve(),
        readiness_timeout_seconds=30,
    )
    source = (FIXTURES / "model.yaml").read_text()
    rendered = ManifestoRenderer(settings).render(source)
    assert rendered.manifesto_digest
    assert rendered.deployment_id.startswith("hard-")
    assert rendered.endpoint.endswith(".models.svc.cluster.local:8000/v1/models")
    assert len(rendered.workloads) == 1
    assert rendered.workloads[0].expected_pods == 1
    assert rendered.workloads[0].pod_selector["app.kubernetes.io/instance"] == (
        rendered.deployment_id
    )
    deployment = next(obj for obj in rendered.objects if obj["kind"] == "Deployment")
    annotations = deployment["spec"]["template"]["metadata"]["annotations"]
    assert annotations[OWNER_ANNOTATION] == rendered.deployment_id
    assert annotations[DIGEST_ANNOTATION] == rendered.intent_digest
    assert all(
        "caller-name-is-replaced" not in resource.name
        for resource in rendered.resources
    )
    assert any(resource.kind == "Deployment" for resource in rendered.resources)


def test_pinned_manifesto_renders_explicit_single_node_lws(tmp_path: Path) -> None:
    pytest.importorskip("manifesto")
    token = tmp_path / "token"
    token.write_text("secret")
    settings = DeploymentSettings(
        cluster_profile=(FIXTURES / "cluster.yaml").resolve(),
        namespace="models",
        bearer_token_file=token.resolve(),
        readiness_timeout_seconds=30,
    )
    source = (
        (FIXTURES / "model.yaml")
        .read_text()
        .replace(
            "  - name: decode\n",
            "  - name: decode\n    workload: leaderworkerset\n",
        )
    )

    rendered = ManifestoRenderer(settings).render(source)

    workload = next(obj for obj in rendered.objects if obj["kind"] == "LeaderWorkerSet")
    assert workload["spec"]["leaderWorkerTemplate"]["size"] == 1
    assert rendered.workloads[0].expected_pods == 1
    assert any(resource.kind == "LeaderWorkerSet" for resource in rendered.resources)


def test_release_placeholder_does_not_select_deployment_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("manifesto")
    token = tmp_path / "token"
    token.write_text("secret")
    settings = DeploymentSettings(
        cluster_profile=(FIXTURES / "cluster.yaml").resolve(),
        namespace="models",
        bearer_token_file=token.resolve(),
        readiness_timeout_seconds=30,
    )
    source = (FIXTURES / "model.yaml").read_text()
    first = ManifestoRenderer(settings).render(source)
    second = ManifestoRenderer(settings).render(
        source.replace("release: caller-name-is-replaced", "release: another-name")
    )
    third = ManifestoRenderer(settings).render(
        source.replace("release: caller-name-is-replaced\n", "")
    )
    assert first.manifesto_digest != second.manifesto_digest
    assert first.intent_digest == second.intent_digest == third.intent_digest
    assert first.deployment_id == second.deployment_id == third.deployment_id
    assert first.objects == second.objects == third.objects
