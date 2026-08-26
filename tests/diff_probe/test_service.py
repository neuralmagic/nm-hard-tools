from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nm_hard_tools.diff_probe.planner import ANNOTATION_PREFIX

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tests.diff_probe.conftest import FakeKubernetes


def test_plan_is_deterministic_and_creates_nothing(
    client: TestClient, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    first = client.post("/v1/probes:plan", json=request_body)
    second = client.post("/v1/probes:plan", json=request_body)
    assert first.status_code == 200
    assert first.json()["probe_id"] == second.json()["probe_id"]
    assert first.json()["plan_sha256"] == second.json()["plan_sha256"]
    assert kube.creates == 0
    varied = client.post("/v1/probes:plan", json={**request_body, "max_tokens": 128})
    assert varied.json()["probe_id"] != first.json()["probe_id"]


def test_plan_renders_a_bounded_job(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    plan = client.post("/v1/probes:plan", json=request_body).json()
    job = plan["resources"][0]
    spec = job["spec"]["template"]["spec"]
    assert job["spec"]["backoffLimit"] == 0
    assert spec["automountServiceAccountToken"] is False
    container = spec["containers"][0]
    assert container["command"][-1] == "nm_hard_tools.diff_probe.worker"
    env = {item["name"] for item in container["env"]}
    assert {"DIFF_PROBE_CONFIG", "OPENAI_API_KEY_A", "OPENAI_API_KEY_B"} <= env
    mounts = {mount["name"] for mount in container["volumeMounts"]}
    assert {"results", "request-sets", "tmp"} <= mounts
    request_sets = next(
        volume for volume in spec["volumes"] if volume["name"] == "request-sets"
    )
    assert request_sets["persistentVolumeClaim"]["readOnly"] is True


def test_unknown_target_and_request_set_are_rejected(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    response = client.post("/v1/probes:plan", json={**request_body, "target_b": "nope"})
    assert response.status_code == 422
    assert "allowed targets" in response.json()["detail"]
    response = client.post(
        "/v1/probes:plan", json={**request_body, "request_set": "nope"}
    )
    assert response.status_code == 422
    assert "allowed request sets" in response.json()["detail"]


def test_smoke_mode_rejects_two_targets_and_diff_requires_two(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    body = {**request_body, "mode": "smoke"}
    assert client.post("/v1/probes:plan", json=body).status_code == 422
    body = {**request_body}
    body.pop("target_b")
    assert client.post("/v1/probes:plan", json=body).status_code == 422


def test_submit_is_idempotent_and_collision_conflicts(
    client: TestClient, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    first = client.post("/v1/probes", json=request_body)
    assert first.status_code == 202
    assert first.json()["state"] == "pending"
    again = client.post("/v1/probes", json=request_body)
    assert again.status_code == 202
    assert kube.creates == 1
    probe = first.json()["probe_id"]
    kube.jobs[probe]["metadata"]["annotations"][f"{ANNOTATION_PREFIX}plan-sha256"] = (
        "not-the-same"
    )
    assert client.post("/v1/probes", json=request_body).status_code == 409


def test_status_follows_job_conditions(
    client: TestClient, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    probe = client.post("/v1/probes", json=request_body).json()["probe_id"]
    kube.jobs[probe]["status"] = {"active": 1}
    assert client.get(f"/v1/probes/{probe}").json()["state"] == "running"
    kube.jobs[probe]["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}]
    }
    assert client.get(f"/v1/probes/{probe}").json()["state"] == "succeeded"
    kube.jobs[probe]["status"] = {
        "conditions": [
            {"type": "Failed", "status": "True", "message": "deadline exceeded"}
        ]
    }
    status = client.get(f"/v1/probes/{probe}").json()
    assert status["state"] == "failed"
    assert status["message"] == "deadline exceeded"


def test_cancel_synthesizes_a_terminal_report(
    client: TestClient, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    probe = client.post("/v1/probes", json=request_body).json()["probe_id"]
    cancelled = client.delete(f"/v1/probes/{probe}")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    report = client.get(f"/v1/probes/{probe}/report")
    assert report.status_code == 200
    assert report.json()["terminal_state"] == "cancelled"
    assert report.json()["error"] == "cancelled by operator"


def test_report_of_a_running_probe_conflicts(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    probe = client.post("/v1/probes", json=request_body).json()["probe_id"]
    assert client.get(f"/v1/probes/{probe}/report").status_code == 409


def test_failed_job_without_worker_output_synthesizes_report(
    client: TestClient, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    probe = client.post("/v1/probes", json=request_body).json()["probe_id"]
    kube.jobs[probe]["status"] = {
        "conditions": [
            {"type": "Failed", "status": "True", "message": "image pull failure"}
        ]
    }
    report = client.get(f"/v1/probes/{probe}/report")
    assert report.status_code == 200
    body = report.json()
    assert body["terminal_state"] == "failed"
    assert body["error"] == "image pull failure"
    assert body["metrics"] == {}
    artifacts = client.get(f"/v1/probes/{probe}/artifacts").json()
    assert {a["name"] for a in artifacts["artifacts"]} >= {
        "effective-configuration.json",
        "report.json",
    }


def test_worker_report_is_served_verbatim(
    client: TestClient,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
    settings: Any,
) -> None:
    plan = client.post("/v1/probes:plan", json=request_body).json()
    probe = plan["probe_id"]
    client.post("/v1/probes", json=request_body)
    kube.jobs[probe]["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}]
    }
    from pathlib import Path

    output = Path(settings.result_root) / probe
    output.mkdir(parents=True)
    durable = {
        "probe_id": probe,
        "terminal_state": "succeeded",
        "mode": "diff",
        "metrics": {"mean_divergence": 0.71, "request_count": 2.0},
        "primary_metric": "mean_divergence",
        "request_count": 2,
        "request_set_fingerprint": {
            "name": "gemma4-vision",
            "file": "x",
            "sha256": "a" * 64,
        },
        "endpoint_identities": plan["endpoint_identities"],
        "model_identity": {"served_model": request_body["model"]},
        "effective_configuration": {},
        "artifact_hashes": {},
        "service_version": "0.1.0",
    }
    (output / "report.json").write_text(json.dumps(durable))
    served = client.get(f"/v1/probes/{probe}/report")
    assert served.status_code == 200
    assert served.json()["metrics"]["mean_divergence"] == 0.71


def test_mutations_require_bearer_auth(
    settings: Any, kube: FakeKubernetes, request_body: dict[str, Any]
) -> None:
    from fastapi.testclient import TestClient

    from nm_hard_tools.diff_probe.api import create_app

    anonymous = TestClient(create_app(settings, kube, "test-token"))
    assert anonymous.post("/v1/probes", json=request_body).status_code == 401
    assert anonymous.post("/v1/probes:plan", json=request_body).status_code == 200
    assert anonymous.get("/v1/probes").status_code == 200
