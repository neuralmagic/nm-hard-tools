from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_hard_tools.evaluation_service.api import create_app
from inference_hard_tools.evaluation_service.config import ServiceSettings
from inference_hard_tools.evaluation_service.models import EvaluationRequest
from inference_hard_tools.evaluation_service.service import EvaluationService

from .conftest import ApiError


class FakeLedger:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[dict[str, Any]] = []

    def register(self, artifact: dict[str, Any]) -> None:
        self.calls.append(artifact)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("ledger unavailable")


if TYPE_CHECKING:
    from tests.evaluation_service.conftest import FakeKubernetes


def test_plan_is_exact_bounded_and_side_effect_free(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    first = service.plan(EvaluationRequest.model_validate(request_body))
    second = service.plan(EvaluationRequest.model_validate(request_body))
    assert first == second
    assert kube.creates == 0
    assert first.endpoint_identity == "http://kimi-k3.default.svc:8000/v1/completions"
    job = first.resources[0]
    assert job["kind"] == "Job"
    assert "nvidia.com/gpu" not in json.dumps(job)
    assert "kueue" not in json.dumps(job).lower()
    assert job["spec"]["backoffLimit"] == 0
    pod_spec = job["spec"]["template"]["spec"]
    assert pod_spec["containers"][0]["command"] == [
        "python",
        "-m",
        "inference_hard_tools.evaluation_service.worker",
    ]
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["imagePullSecrets"] == [{"name": "registry-pull"}]
    assert pod_spec["nodeSelector"] == {"kubernetes.io/arch": "amd64"}
    assert pod_spec["securityContext"]["fsGroup"] == 65532
    env = {item["name"]: item for item in pod_spec["containers"][0]["env"]}
    assert env["HF_HOME"]["value"] == "/tmp/huggingface"
    assert env["HF_DATASETS_CACHE"]["value"] == "/tmp/huggingface/datasets"
    assert "@sha256:" in pod_spec["containers"][0]["image"]
    assert "LM_EVAL_LEDGER_WRITER_TOKEN" not in env
    worker_config = json.loads(env["LM_EVAL_EVALUATION_CONFIG"]["value"])
    assert set(worker_config) == {
        "effective_configuration",
        "endpoint_identity",
        "evaluation_id",
        "invocation",
        "request",
    }
    assert job["metadata"]["labels"]["app.kubernetes.io/instance"] == "lm-eval-test"
    assert job["metadata"]["labels"]["evaluation.lm-eval.io/profile"] == "gsm8k"


def test_plan_identity_includes_resolved_operator_configuration(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    request = EvaluationRequest.model_validate(request_body)
    first = EvaluationService(settings, kube).plan(request)
    changed = settings.model_copy(
        update={"worker_image": "registry.example/lm-eval@sha256:" + "b" * 64}
    )
    second = EvaluationService(changed, kube).plan(request)
    assert first.request_sha256 == second.request_sha256
    assert first.plan_sha256 != second.plan_sha256
    assert first.evaluation_id != second.evaluation_id
    moved = settings.model_copy(update={"result_root": settings.result_root + "-moved"})
    third = EvaluationService(moved, kube).plan(request)
    assert first.plan_sha256 != third.plan_sha256
    assert first.effective_configuration["lm_eval_invocation"]["tasks"][0]["dataset"][
        "revision"
    ]

    profile = settings.profiles["gsm8k"]
    task = profile.tasks[0]
    changed_task = task.model_copy(
        update={"dataset": task.dataset.model_copy(update={"revision": "b" * 40})}
    )
    changed_profile = profile.model_copy(update={"tasks": [changed_task]})
    reprofiled = settings.model_copy(update={"profiles": {"gsm8k": changed_profile}})
    fourth = EvaluationService(reprofiled, kube).plan(request)
    assert fourth.request_sha256 == first.request_sha256
    assert fourth.plan_sha256 != first.plan_sha256


def test_profile_defaults_bounds_and_unknown_profile(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    defaulted = dict(request_body)
    defaulted.pop("num_fewshot")
    defaulted.pop("generation_limit")
    effective = service.plan(
        EvaluationRequest.model_validate(defaulted)
    ).effective_configuration
    assert effective["resolved_num_fewshot"] == 5
    assert effective["resolved_generation_limit"] == 512

    too_large = {**request_body, "limit": 1320}
    with pytest.raises(ValueError, match="profile maximum 1319"):
        service.plan(EvaluationRequest.model_validate(too_large))
    unknown = {**request_body, "profile": "not-installed"}
    with pytest.raises(ValueError, match="allowed profiles: gsm8k"):
        service.plan(EvaluationRequest.model_validate(unknown))

    profile = settings.profiles["gsm8k"]
    missing_task = profile.tasks[0].model_copy(
        update={"config_path": "lm_eval/tasks/missing/task.yaml"}
    )
    missing_profile = profile.model_copy(update={"tasks": [missing_task]})
    missing_settings = settings.model_copy(
        update={"profiles": {"gsm8k": missing_profile}}
    )
    with pytest.raises(ValueError, match="missing installed task"):
        EvaluationService(missing_settings, kube).plan(
            EvaluationRequest.model_validate(request_body)
        )


def test_shared_dataset_cache_is_offline_read_only_and_profile_scoped(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    profile = settings.profiles["gsm8k"].model_copy(
        update={"dataset_token_secret_name": "hf-dataset-token"}
    )
    cached = settings.model_copy(
        update={
            "dataset_cache_claim_name": "eval-datasets",
            "profiles": {"gsm8k": profile},
        }
    )
    plan = EvaluationService(cached, kube).plan(
        EvaluationRequest.model_validate(request_body)
    )
    pod = plan.resources[0]["spec"]["template"]["spec"]
    env = {item["name"]: item for item in pod["containers"][0]["env"]}
    assert env["HF_HUB_OFFLINE"]["value"] == "1"
    assert env["HF_DATASETS_OFFLINE"]["value"] == "1"
    assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == ("hf-dataset-token")
    cache_mount = next(
        item
        for item in pod["containers"][0]["volumeMounts"]
        if item["name"] == "dataset-cache"
    )
    assert cache_mount["readOnly"] is True
    cache_volume = next(
        item for item in pod["volumes"] if item["name"] == "dataset-cache"
    )
    assert cache_volume["persistentVolumeClaim"]["readOnly"] is True


def test_global_hf_token_fallback_and_profile_override(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    global_settings = settings.model_copy(update={"hf_token_secret_name": "hf-secret"})
    global_plan = EvaluationService(global_settings, kube).plan(
        EvaluationRequest.model_validate(request_body)
    )
    env = {
        item["name"]: item
        for item in global_plan.resources[0]["spec"]["template"]["spec"]["containers"][
            0
        ]["env"]
    }
    assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "hf-secret",
        "key": "HF_TOKEN",
    }
    assert global_plan.effective_configuration["dataset_auth_secret_ref"] == {
        "name": "hf-secret",
        "key": "HF_TOKEN",
    }

    profile = settings.profiles["gsm8k"].model_copy(
        update={
            "dataset_token_secret_name": "profile-token",
            "dataset_token_secret_key": "token",
        }
    )
    overridden = global_settings.model_copy(update={"profiles": {"gsm8k": profile}})
    override_plan = EvaluationService(overridden, kube).plan(
        EvaluationRequest.model_validate(request_body)
    )
    override_env = {
        item["name"]: item
        for item in override_plan.resources[0]["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
    }
    assert override_env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "profile-token",
        "key": "token",
    }
    assert override_plan.plan_sha256 != global_plan.plan_sha256


def test_submit_status_cancel_and_restart_reconstruction(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    request = EvaluationRequest.model_validate(request_body)
    service = EvaluationService(settings, kube)
    submitted = service.submit(request)
    assert submitted.state == "pending"
    assert service.submit(request).evaluation_id == submitted.evaluation_id
    assert kube.creates == 1

    kube.jobs[submitted.evaluation_id]["status"] = {
        "active": 1,
        "startTime": "2026-08-17T12:00:00Z",
    }
    restarted = EvaluationService(settings, kube)
    assert restarted.get(submitted.evaluation_id).state == "running"
    assert restarted.list().evaluations[0].evaluation_id == submitted.evaluation_id
    restarted._synthesize_terminal_report(
        kube.get_job(settings.namespace, submitted.evaluation_id),
        "failed",
        "stale attempt failure",
    )
    stale = json.loads(
        (
            Path(settings.result_root) / submitted.evaluation_id / "report.json"
        ).read_text()
    )
    assert stale["terminal_state"] == "failed"
    try:
        restarted.get_report(submitted.evaluation_id)
    except Exception as exc:  # noqa: BLE001
        assert "not terminal" in str(exc)
    else:
        raise AssertionError("running evaluation exposed a stale report")
    output = Path(settings.result_root) / submitted.evaluation_id
    for path in output.iterdir():
        path.unlink()
    output.rmdir()
    cancelled = restarted.cancel(submitted.evaluation_id)
    assert cancelled.state == "cancelled"
    assert restarted.cancel(submitted.evaluation_id).state == "cancelled"
    report = restarted.get_report(submitted.evaluation_id)
    assert report.terminal_state == "cancelled"
    assert report.effective_configuration["profile"] == "gsm8k"
    assert restarted.list_artifacts(submitted.evaluation_id).artifacts


def test_synthesized_artifact_manifest_is_repaired(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    job = kube.jobs[evaluation]
    job["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    service._synthesize_terminal_report(job, "failed", "worker disappeared")
    output = Path(settings.result_root) / evaluation
    manifest = output / "artifacts.json"
    manifest.unlink()

    service._synthesize_terminal_report(job, "failed", "worker disappeared")
    repaired = json.loads(manifest.read_text())
    assert {item["name"] for item in repaired["artifacts"]} == {
        "effective-configuration.json",
        "report.json",
    }

    manifest.write_text("{partial", encoding="utf-8")
    assert service.list_artifacts(evaluation).artifacts
    assert not list(output.glob(".artifacts-*"))


def test_result_root_must_be_below_filesystem_root(
    settings: ServiceSettings,
) -> None:
    data = settings.model_dump()
    data["result_root"] = "/"
    with pytest.raises(ValidationError, match="below the root"):
        ServiceSettings.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "https://huggingface.co/datasets/openai/gsm8k"),
        ("revision", "main"),
    ],
)
def test_profile_datasets_require_repository_ids_and_immutable_revisions(
    settings: ServiceSettings, field: str, value: str
) -> None:
    data = settings.model_dump()
    data["profiles"]["gsm8k"]["tasks"][0]["dataset"][field] = value
    with pytest.raises(ValidationError):
        ServiceSettings.model_validate(data)


def test_terminal_report_registers_measured_ledger_pointer_once(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    ledger = FakeLedger()
    service = EvaluationService(settings, kube, ledger)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    kube.jobs[evaluation]["status"] = {
        "conditions": [{"type": "Failed", "status": "True"}]
    }

    service.get_report(evaluation)
    assert ledger.calls == []
    service.reconcile_ledger()
    service.reconcile_ledger()

    assert len(ledger.calls) == 1
    artifact = ledger.calls[0]
    assert artifact["workstream_id"] == "ws-0123456789ab"
    assert artifact["kind"] == "lm-evaluation-report"
    assert artifact["operation_ref"] == evaluation
    assert "commit_sha" not in artifact
    assert artifact["storage_url"].endswith(f"/{evaluation}/report.json")
    marker = Path(settings.result_root) / evaluation / "ledger-registration.json"
    assert json.loads(marker.read_text())["artifact_id"] == artifact["artifact_id"]
    assert kube.list_calls[-1] == (
        "evals",
        (
            "app.kubernetes.io/managed-by=lm-eval-service,"
            "app.kubernetes.io/instance=lm-eval-test"
        ),
        None,
    )


def test_ledger_failure_never_blocks_report_and_retries(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    ledger = FakeLedger(failures=1)
    service = EvaluationService(settings, kube, ledger)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    kube.jobs[evaluation]["status"] = {
        "conditions": [{"type": "Failed", "status": "True"}]
    }

    assert service.get_report(evaluation).terminal_state == "failed"
    service.reconcile_ledger()
    assert service.get_report(evaluation).terminal_state == "failed"
    service.reconcile_ledger()
    assert len(ledger.calls) == 2


def test_cancel_race_preserves_completed_job_and_worker_report(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    output = Path(settings.result_root) / evaluation
    original_cancel = kube.cancel_job
    raced = False

    def complete_during_cancel(
        namespace: str, name: str, cancelled_at: str, resource_version: str
    ) -> dict[str, Any]:
        nonlocal raced
        if not raced:
            raced = True
            job = kube.jobs[name]
            job["metadata"]["resourceVersion"] = "2"
            job["status"] = {
                "startTime": "2026-08-17T12:00:00Z",
                "completionTime": "2026-08-17T12:00:05Z",
                "conditions": [{"type": "Complete", "status": "True"}],
            }
            service._synthesize_terminal_report(job, "succeeded", "")
            raise ApiError(409)
        return original_cancel(namespace, name, cancelled_at, resource_version)

    kube.cancel_job = complete_during_cancel  # type: ignore[method-assign]
    status = service.cancel(evaluation)
    assert status.state == "succeeded"
    report = json.loads((output / "report.json").read_text())
    assert report["terminal_state"] == "succeeded"
    assert "cancelled-at" not in kube.jobs[evaluation]["metadata"]["annotations"]


def test_worker_report_wins_if_published_while_cancellation_settles(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    job = kube.jobs[evaluation]
    job["status"] = {"active": 1, "startTime": "2026-08-17T12:00:00Z"}

    def publish_worker_report() -> None:
        service._synthesize_terminal_report(job, "succeeded", "")

    kube.quiescence_hook = publish_worker_report
    status = service.cancel(evaluation)
    assert status.state == "succeeded"
    assert service.get_report(evaluation).terminal_state == "succeeded"


def test_failed_job_uses_terminal_condition_timestamp(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    service = EvaluationService(settings, kube)
    evaluation = service.submit(
        EvaluationRequest.model_validate(request_body)
    ).evaluation_id
    job = kube.jobs[evaluation]
    job["status"] = {
        "startTime": "2026-08-17T12:00:00Z",
        "conditions": [
            {
                "type": "Failed",
                "status": "True",
                "lastTransitionTime": "2026-08-17T12:00:07Z",
                "reason": "DeadlineExceeded",
            }
        ],
    }
    report = service.get_report(evaluation)
    assert report.measurement_completed_at.isoformat() == "2026-08-17T12:00:07+00:00"
    assert report.latency_seconds == 7


def test_unknown_fields_and_ssrf_are_rejected(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    with_url = {**request_body, "endpoint": "http://169.254.169.254/latest/meta-data"}
    response = client.post("/v1/evaluations:plan", json=with_url)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"

    arbitrary_target = {**request_body, "target": "http://127.0.0.1:8000"}
    response = client.post("/v1/evaluations:plan", json=arbitrary_target)
    assert response.status_code == 422

    unknown_target = {**request_body, "target": "not-allowed"}
    response = client.post("/v1/evaluations:plan", json=unknown_target)
    assert response.status_code == 422
    assert "allowed targets" in response.json()["detail"]


def test_result_response_bound(
    client: TestClient,
    kube: FakeKubernetes,
    settings: ServiceSettings,
    request_body: dict[str, Any],
) -> None:
    submitted = client.post("/v1/evaluations", json=request_body).json()
    output = Path(settings.result_root) / submitted["evaluation_id"]
    output.mkdir()
    (output / "report.json").write_bytes(b"{" + b"x" * 262_145)
    kube.jobs[submitted["evaluation_id"]]["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}]
    }
    response = client.get(f"/v1/evaluations/{submitted['evaluation_id']}/report")
    assert response.status_code == 409
    assert "exceeds" in response.json()["detail"]


def test_logs_are_bounded_and_tail_is_validated(
    client: TestClient,
    kube: FakeKubernetes,
    settings: ServiceSettings,
    request_body: dict[str, Any],
) -> None:
    evaluation = client.post("/v1/evaluations", json=request_body).json()[
        "evaluation_id"
    ]
    kube.logs = "x" * 100_000
    response = client.get(f"/v1/evaluations/{evaluation}/logs?tail_lines=200")
    assert response.status_code == 200
    assert len(response.text) == 65_536
    assert (
        client.get(f"/v1/evaluations/{evaluation}/logs?tail_lines=201").status_code
        == 422
    )
    kube.logs = ""
    output = Path(settings.result_root) / evaluation
    output.mkdir(exist_ok=True)
    (output / "worker.log").write_text("one\ntwo\nthree\n")
    response = client.get(f"/v1/evaluations/{evaluation}/logs?tail_lines=2")
    assert response.text == "two\nthree"
    artifacts = client.get(f"/v1/evaluations/{evaluation}/artifacts").json()
    assert "worker.log" in {item["name"] for item in artifacts["artifacts"]}


def test_mutations_require_auth_and_are_rate_limited(
    settings: ServiceSettings,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    limited = settings.model_copy(update={"mutation_rate_limit_per_minute": 1})
    client = TestClient(create_app(limited, kube, "operator-token"))
    assert client.post("/v1/evaluations", json=request_body).status_code == 401
    headers = {"Authorization": "Bearer operator-token"}
    assert (
        client.post("/v1/evaluations", headers=headers, json=request_body).status_code
        == 202
    )
    second = {**request_body, "model": "different-model"}
    response = client.post("/v1/evaluations", headers=headers, json=second)
    assert response.status_code == 429


def test_auth_and_body_limit_run_before_request_parsing(
    settings: ServiceSettings, kube: FakeKubernetes
) -> None:
    client = TestClient(create_app(settings, kube, "operator-token"))
    oversized = b"{" + b"x" * 1_048_576
    unauthorized = client.post(
        "/v1/evaluations",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert unauthorized.status_code == 401
    rejected = client.post(
        "/v1/evaluations",
        content=oversized,
        headers={
            "Authorization": "Bearer operator-token",
            "Content-Type": "application/json",
        },
    )
    assert rejected.status_code == 413
    plan = client.post(
        "/v1/evaluations:plan",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert plan.status_code == 413
