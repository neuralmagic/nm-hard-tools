"""Probe lifecycle built directly on Kubernetes Jobs — the operation record."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from nm_hard_tools.diff_probe import SERVICE_VERSION
from nm_hard_tools.diff_probe.artifacts import (
    MAX_REPORT_BYTES,
    ArtifactConflict,
    ProbeArtifactStore,
    artifact_metadata,
    atomic_json,
)
from nm_hard_tools.diff_probe.kube import CANCELLED_AT_ANNOTATION
from nm_hard_tools.diff_probe.models import (
    ProbeArtifactList,
    ProbeList,
    ProbePlanResponse,
    ProbeReport,
    ProbeRequest,
    ProbeStatus,
    validate_probe_id,
)
from nm_hard_tools.diff_probe.planner import ANNOTATION_PREFIX, plan_probe

if TYPE_CHECKING:
    from nm_hard_tools.diff_probe.config import ProbeServiceSettings
    from nm_hard_tools.diff_probe.kube import KubernetesBackend

MANAGED_LABEL = "app.kubernetes.io/managed-by=diff-probe-service"


class ProbeNotFound(KeyError):
    pass


class ProbeConflict(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _field(value: dict[str, Any], snake: str, camel: str | None = None) -> Any:
    return value.get(snake, value.get(camel or snake))


class ProbeService:
    def __init__(self, settings: ProbeServiceSettings, kube: KubernetesBackend) -> None:
        self.settings = settings
        self.kube = kube
        self.artifacts = ProbeArtifactStore(settings.result_root)

    @property
    def managed_selector(self) -> str:
        return (
            f"{MANAGED_LABEL},"
            f"app.kubernetes.io/instance={self.settings.service_instance}"
        )

    def plan(self, request: ProbeRequest) -> ProbePlanResponse:
        return plan_probe(self.settings, request)

    def submit(self, request: ProbeRequest) -> ProbeStatus:
        plan = self.plan(request)
        try:
            job = self.kube.create_job(self.settings.namespace, plan.resources[0])
        except Exception as exc:
            if getattr(exc, "status", None) != 409:
                raise
            job = self.kube.get_job(self.settings.namespace, plan.probe_id)
            existing = self._annotations(job).get(f"{ANNOTATION_PREFIX}plan-sha256")
            if existing != plan.plan_sha256:
                raise ProbeConflict("probe ID collision") from exc
        return self._status(job)

    def list(self, limit: int = 50) -> ProbeList:
        jobs = self.kube.list_jobs(
            self.settings.namespace, self.managed_selector, limit
        )
        statuses = [self._status(job) for job in jobs]
        statuses.sort(
            key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return ProbeList(probes=statuses[:limit])

    def get(self, probe: str) -> ProbeStatus:
        validate_probe_id(probe)
        try:
            job = self.kube.get_job(self.settings.namespace, probe)
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                raise ProbeNotFound(probe) from exc
            raise
        return self._status(job)

    def cancel(self, probe: str) -> ProbeStatus:
        validate_probe_id(probe)
        for _attempt in range(3):
            try:
                job = self.kube.get_job(self.settings.namespace, probe)
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    raise ProbeNotFound(probe) from exc
                raise
            current = self._status(job)
            if current.state in {"succeeded", "failed"}:
                return current
            if current.state == "cancelled":
                reconciled = self._quiesce(probe)
                if self._status(reconciled).state == "cancelled":
                    self._synthesize_terminal_report(
                        reconciled, "cancelled", "cancelled by operator"
                    )
                return self._status(reconciled)
            report_path = self.artifacts.path(probe, "report.json")
            if report_path.exists():
                ProbeReport.model_validate(
                    self.artifacts.read_json(report_path, MAX_REPORT_BYTES)
                )
                raise ProbeConflict(
                    "a terminal worker report exists while Kubernetes status settles"
                )
            metadata = job.get("metadata") or {}
            resource_version = _field(metadata, "resource_version", "resourceVersion")
            if not resource_version:
                raise ProbeConflict("Kubernetes Job has no resourceVersion")
            try:
                self.kube.cancel_job(
                    self.settings.namespace, probe, _now(), str(resource_version)
                )
            except Exception as exc:
                if getattr(exc, "status", None) == 409:
                    continue
                raise
            reconciled = self._quiesce(probe)
            if self._status(reconciled).state == "cancelled":
                self._synthesize_terminal_report(
                    reconciled, "cancelled", "cancelled by operator"
                )
            return self._status(reconciled)
        raise ProbeConflict("probe changed concurrently; retry cancellation")

    def list_artifacts(self, probe: str) -> ProbeArtifactList:
        self.get(probe)
        return self.artifacts.list(probe)

    def get_report(self, probe: str) -> ProbeReport:
        status = self.get(probe)
        path = self.artifacts.path(probe, "report.json")
        if status.state not in {"succeeded", "failed", "cancelled"}:
            raise ProbeConflict("probe is not terminal")
        if not path.exists():
            if status.state in {"failed", "cancelled"}:
                job = self.kube.get_job(self.settings.namespace, probe)
                if status.state == "cancelled":
                    job = self._quiesce(probe)
                self._synthesize_terminal_report(
                    job,
                    status.state,
                    status.message or f"Kubernetes Job {status.state}",
                )
            else:
                raise ProbeNotFound(f"report for {probe}")
        report = ProbeReport.model_validate(
            self.artifacts.read_json(path, MAX_REPORT_BYTES)
        )
        if report.terminal_state != status.state:
            raise ProbeConflict(
                "durable report terminal state does not match the Kubernetes Job"
            )
        return report

    def logs(self, probe: str, tail_lines: int) -> str:
        self.get(probe)
        try:
            logs = self.kube.read_logs(self.settings.namespace, probe, tail_lines)
        except Exception:  # noqa: BLE001 - durable artifacts outlive Pod loss.
            logs = ""
        return logs[-65_536:] if logs else ""

    def ready(self) -> bool:
        return self.kube.ready(self.settings.namespace) and self.artifacts.ready()

    def _quiesce(self, probe: str) -> dict[str, Any]:
        try:
            return self.kube.wait_for_job_quiescence(
                self.settings.namespace, probe, 10.0
            )
        except TimeoutError as exc:
            raise ProbeConflict(str(exc)) from exc

    def _synthesize_terminal_report(
        self, job: dict[str, Any], state: str, error: str
    ) -> None:
        """Persist a bounded report when Kubernetes terminates before worker output."""
        status = self._status(job)
        output_dir = self.artifacts.output_dir(status.probe_id)
        output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
        report_path = output_dir / "report.json"
        if report_path.exists():
            self.artifacts.ensure_manifest(output_dir, status.probe_id)
            return
        annotations = self._annotations(job)
        effective = json.loads(
            annotations[f"{ANNOTATION_PREFIX}effective-configuration"]
        )
        config_path = output_dir / "effective-configuration.json"
        atomic_json(config_path, effective)
        config_meta = artifact_metadata(config_path, "application/json")
        request = json.loads(annotations[f"{ANNOTATION_PREFIX}request"])
        report = ProbeReport(
            probe_id=status.probe_id,
            terminal_state=state,
            mode=request["mode"],
            metrics={},
            primary_metric=(
                "mean_divergence" if request["mode"] == "diff" else "pass_rate"
            ),
            request_count=0,
            request_set_fingerprint={
                "name": request["request_set"],
                "file": effective["request_set_configuration"]["file"],
                "sha256": effective["request_set_configuration"]["sha256"],
            },
            endpoint_identities=status.endpoint_identities,
            model_identity={"served_model": status.model},
            effective_configuration=effective,
            measurement_started_at=status.started_at,
            measurement_completed_at=status.completed_at,
            artifact_hashes={config_meta["name"]: config_meta["sha256"]},
            service_version=SERVICE_VERSION,
            error=error[:4096],
        )
        try:
            atomic_json(report_path, report.model_dump(mode="json"))
        except OSError:
            return
        self.artifacts.ensure_manifest(output_dir, status.probe_id)

    @staticmethod
    def _annotations(job: dict[str, Any]) -> dict[str, str]:
        return _field(job.get("metadata", {}), "annotations") or {}

    def _status(self, job: dict[str, Any]) -> ProbeStatus:
        metadata = job.get("metadata", {})
        status = job.get("status") or {}
        annotations = self._annotations(job)
        spec = job.get("spec") or {}
        state = "pending"
        message = None
        terminal_condition = None
        durable_completed_at = None
        for condition in status.get("conditions") or []:
            if str(condition.get("status")).lower() != "true":
                continue
            condition_type = str(condition.get("type", "")).lower()
            if condition_type == "complete":
                state = "succeeded"
                terminal_condition = condition
            elif condition_type == "failed":
                state = "failed"
                terminal_condition = condition
                message = condition.get("message") or condition.get("reason")
        if state == "pending" and CANCELLED_AT_ANNOTATION in annotations:
            state = "cancelled"
            message = "cancelled by operator"
        if state == "pending" and (status.get("active") or 0) > 0:
            state = "running"
        if state == "pending" and _field(spec, "suspend") is True:
            state = "cancelled"
        if state == "cancelled":
            report_path = self.artifacts.path(metadata["name"], "report.json")
            if report_path.exists():
                try:
                    durable = ProbeReport.model_validate(
                        self.artifacts.read_json(report_path, MAX_REPORT_BYTES)
                    )
                except (OSError, ValueError, ProbeConflict, ArtifactConflict):
                    pass
                else:
                    state = durable.terminal_state
                    message = durable.error
                    durable_completed_at = durable.measurement_completed_at
        completed_at = _field(status, "completion_time", "completionTime")
        if completed_at is None and terminal_condition is not None:
            completed_at = _field(
                terminal_condition, "last_transition_time", "lastTransitionTime"
            )
        if completed_at is None and durable_completed_at is not None:
            completed_at = durable_completed_at
        request = json.loads(annotations[f"{ANNOTATION_PREFIX}request"])
        return ProbeStatus(
            probe_id=metadata["name"],
            state=state,
            created_at=_field(metadata, "creation_timestamp", "creationTimestamp"),
            started_at=_field(status, "start_time", "startTime"),
            completed_at=completed_at,
            mode=request["mode"],
            model=request["model"],
            request_set=request["request_set"],
            endpoint_identities=json.loads(
                annotations[f"{ANNOTATION_PREFIX}endpoint-identities"]
            ),
            request_sha256=annotations[f"{ANNOTATION_PREFIX}request-sha256"],
            message=message,
        )
