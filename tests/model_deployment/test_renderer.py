from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.renderer import (
    DIGEST_ANNOTATION,
    KUEUE_LWS_NAME_MAX,
    OWNER_ANNOTATION,
    ManifestoConfigError,
    ManifestoRenderer,
    NameBudget,
    OperatorConfigurationError,
    RenderedDeployment,
    _validate_lws_name_fits_kueue_label,
    parse_manifesto_config,
)

FIXTURES = Path(__file__).parent / "fixtures"

# The rendered LeaderWorkerSet name is a function of these operator knobs, the
# derived deployment identity, and the caller role name.
OPERATOR_PROFILE_NAMES = ["ci", "pirate-tms-hard-tools-backport"]


def operator_cluster_profile(
    tmp_path: Path,
    *,
    profile_name: str = "test-stateless-b200",
    user_prefix: bool = False,
    local_queue: str | None = None,
) -> Path:
    profile = yaml.safe_load((FIXTURES / "cluster.yaml").read_text())
    profile["name"] = profile_name
    profile["naming"] = {"user_prefix": user_prefix}
    if local_queue is not None:
        profile["kueue"] = {"local_queue": local_queue}
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return path


def renderer_for(tmp_path: Path, cluster_profile: Path) -> ManifestoRenderer:
    token = tmp_path / "token"
    token.write_text("secret")
    return ManifestoRenderer(
        DeploymentSettings(
            cluster_profile=cluster_profile.resolve(),
            namespace="models",
            bearer_token_file=token.resolve(),
            readiness_timeout_seconds=30,
        )
    )


def lws_source(role_name: str, *, replicas: int = 1) -> str:
    return (
        (FIXTURES / "model.yaml")
        .read_text()
        .replace(
            "  - name: decode\n    lws: {size: 1}\n",
            f"  - name: {role_name}\n    workload: leaderworkerset\n"
            f"    lws: {{size: 1, replicas: {replicas}}}\n",
        )
    )


def lws_name(rendered: RenderedDeployment) -> str:
    workload = next(obj for obj in rendered.objects if obj["kind"] == "LeaderWorkerSet")
    return str(workload["metadata"]["name"])


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
    assert len(rendered.deployment_id) == len("hard-") + 24
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
    assert len(workload["metadata"]["name"].encode()) <= KUEUE_LWS_NAME_MAX
    assert "nm-hard-tools-" not in workload["metadata"]["name"]
    kueue_workload_label = f"leaderworkerset-{workload['metadata']['name']}-0-00000"
    assert len(kueue_workload_label.encode()) <= 63
    assert rendered.workloads[0].expected_pods == 1
    assert any(resource.kind == "LeaderWorkerSet" for resource in rendered.resources)


@pytest.mark.parametrize("profile_name", OPERATOR_PROFILE_NAMES)
@pytest.mark.parametrize("user_prefix", [False, True])
@pytest.mark.parametrize("local_queue", ["models-queue", None])
def test_operator_naming_knobs_keep_the_lws_name_within_the_kueue_limit(
    tmp_path: Path, profile_name: str, user_prefix: bool, local_queue: str | None
) -> None:
    """No operator profile may make an ordinary role render an unusable name."""

    pytest.importorskip("manifesto")
    renderer = renderer_for(
        tmp_path,
        operator_cluster_profile(
            tmp_path,
            profile_name=profile_name,
            user_prefix=user_prefix,
            local_queue=local_queue,
        ),
    )

    name = lws_name(renderer.render(lws_source("decode")))

    assert len(name.encode()) <= KUEUE_LWS_NAME_MAX
    assert len(f"leaderworkerset-{name}-0-00000".encode()) <= 63


@pytest.mark.parametrize("user_prefix", [False, True])
@pytest.mark.parametrize("local_queue", ["models-queue", None])
@pytest.mark.parametrize("overshoot", [-1, 0, 1])
def test_lws_name_is_label_safe_or_rejected_before_any_mutation(
    tmp_path: Path, user_prefix: bool, local_queue: str | None, overshoot: int
) -> None:
    """Every rendered name either fits the Kueue budget or is rejected.

    The operator-derived prefix is measured from a one-character role name so
    the boundary is exercised exactly at the limit and one character either
    side without restating how the renderer composes names. `local_queue` is a
    knob because whether Kueue manages the namespace is a property of the
    cluster, not of the rendered object: an unlabelled render is no proof that
    the name will not become a Kueue Workload label value.
    """

    pytest.importorskip("manifesto")
    renderer = renderer_for(
        tmp_path,
        operator_cluster_profile(
            tmp_path, user_prefix=user_prefix, local_queue=local_queue
        ),
    )
    operator_prefix = len(lws_name(renderer.render(lws_source("d")))) - 1
    role_name = "d" * (KUEUE_LWS_NAME_MAX - operator_prefix + overshoot)

    if overshoot > 0:
        with pytest.raises(
            (ManifestoConfigError, OperatorConfigurationError)
        ) as rejected:
            renderer.render(lws_source(role_name))
        assert str(KUEUE_LWS_NAME_MAX) in str(rejected.value)
        return

    name = lws_name(renderer.render(lws_source(role_name)))
    assert len(name.encode()) == KUEUE_LWS_NAME_MAX + overshoot


@pytest.mark.parametrize("replicas", [1, 10, 11, 101])
def test_lws_name_budget_leaves_room_for_the_widest_group_index(
    tmp_path: Path, replicas: int
) -> None:
    """More replicas widen the group index Kueue appends, so they cost budget.

    Kueue names the Workload for group `i` `leaderworkerset-<name>-<i>-<hash>`,
    so a LeaderWorkerSet that reaches a two-digit group index needs a name one
    character shorter than a single-replica one. The budget here is derived from
    that format rather than from the renderer, so an implementation that ignores
    the replica count fails on the accepted case.
    """

    pytest.importorskip("manifesto")
    renderer = renderer_for(tmp_path, operator_cluster_profile(tmp_path))
    operator_prefix = len(lws_name(renderer.render(lws_source("d")))) - 1
    widest_group_index = str(replicas - 1)
    limit = 63 - len(f"leaderworkerset--{widest_group_index}-00000")

    accepted = lws_name(
        renderer.render(lws_source("d" * (limit - operator_prefix), replicas=replicas))
    )

    assert len(accepted.encode()) == limit
    assert len(f"leaderworkerset-{accepted}-{widest_group_index}-00000".encode()) <= 63
    with pytest.raises(ManifestoConfigError):
        renderer.render(
            lws_source("d" * (limit - operator_prefix + 1), replicas=replicas)
        )


def test_over_long_name_names_the_caller_field_that_can_be_shortened(
    tmp_path: Path,
) -> None:
    pytest.importorskip("manifesto")
    renderer = renderer_for(tmp_path, operator_cluster_profile(tmp_path))

    with pytest.raises(ManifestoConfigError) as rejected:
        renderer.render(lws_source("decode-with-a-very-long-role-name"))

    assert rejected.value.field == "roles[0].name"
    assert "must be at most 39 characters" in str(rejected.value)
    assert "Shorten roles[0].name" in str(rejected.value)


def test_live_incident_user_prefixed_lws_name_is_operator_configuration() -> None:
    """The 2026-09-01 incident name, which no caller field could shorten.

    A profile with `naming: {user_prefix: true}` rendered
    `nm-hard-tools-hard-5123293dffec656c9d7c5f26-decode`; the operator-derived
    prefix alone is 44 characters, so the caller has no field to correct.
    """

    incident = "nm-hard-tools-hard-5123293dffec656c9d7c5f26-decode"
    assert len(incident) == 50

    with pytest.raises(OperatorConfigurationError, match="naming.user_prefix"):
        _validate_lws_name_fits_kueue_label(
            {"kind": "LeaderWorkerSet", "metadata": {"name": incident}},
            {
                incident: NameBudget(
                    len("nm-hard-tools-hard-5123293dffec656c9d7c5f26-"), "roles[0].name"
                )
            },
        )


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
