"""Evaluation lifecycle built directly on Kubernetes Jobs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_hard_tools.evaluation_service import lm_eval_commit
from inference_hard_tools.evaluation_service.artifacts import (
    MAX_REPORT_BYTES,
    ArtifactConflict,
    ArtifactStore,
    artifact_metadata,
    atomic_json,
)
from inference_hard_tools.evaluation_service.models import (
    ArtifactList,
    EvaluationList,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    PlanResponse,
    validate_evaluation_id,
)
from inference_hard_tools.evaluation_service.planner import (
    ANNOTATION_PREFIX,
    plan_evaluation,
)

if TYPE_CHECKING:
    from inference_hard_tools.evaluation_service.config import ServiceSettings
    from inference_hard_tools.evaluation_service.kube import KubernetesBackend
    from inference_hard_tools.evaluation_service.ledger_client import LedgerRegistrar


MANAGED_LABEL = "app.kubernetes.io/managed-by=lm-eval-service"
logger = logging.getLogger(__name__)


class EvaluationNotFound(KeyError):
    pass


class EvaluationConflict(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _field(value: dict[str, Any], snake: str, camel: str | None = None) -> Any:
    return value.get(snake, value.get(camel or snake))


class EvaluationService:
    def __init__(
        self,
        settings: ServiceSettings,
        kube: KubernetesBackend,
        ledger: LedgerRegistrar | None = None,
    ) -> None:
        self.settings = settings
        self.kube = kube
        self.ledger = ledger
        self.lm_eval_revision = lm_eval_commit()
        self.artifacts = ArtifactStore(settings.result_root)

    @property
    def managed_selector(self) -> str:
        return (
            f"{MANAGED_LABEL},"
            f"app.kubernetes.io/instance={self.settings.service_instance}"
        )

    def plan(self, request: EvaluationRequest) -> PlanResponse:
        return plan_evaluation(self.settings, request, self.lm_eval_revision)

    def submit(self, request: EvaluationRequest) -> EvaluationStatus:
        plan = self.plan(request)
        try:
            job = self.kube.create_job(self.settings.namespace, plan.resources[0])
        except Exception as exc:
            if getattr(exc, "status", None) != 409:
                raise
            job = self.kube.get_job(self.settings.namespace, plan.evaluation_id)
            existing = self._annotations(job).get(f"{ANNOTATION_PREFIX}plan-sha256")
            if existing != plan.plan_sha256:
                raise EvaluationConflict("evaluation ID collision") from exc
        return self._status(job)

    def list(self, limit: int = 50) -> EvaluationList:
        jobs = self.kube.list_jobs(
            self.settings.namespace, self.managed_selector, limit
        )
        statuses = [self._status(job) for job in jobs]
        statuses.sort(
            key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return EvaluationList(evaluations=statuses[:limit])

    def get(self, evaluation: str) -> EvaluationStatus:
        validate_evaluation_id(evaluation)
        try:
            job = self.kube.get_job(self.settings.namespace, evaluation)
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                raise EvaluationNotFound(evaluation) from exc
            raise
        return self._status(job)

    def cancel(self, evaluation: str) -> EvaluationStatus:
        validate_evaluation_id(evaluation)
        for _attempt in range(3):
            try:
                job = self.kube.get_job(self.settings.namespace, evaluation)
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    raise EvaluationNotFound(evaluation) from exc
                raise
            current = self._status(job)
            if current.state in {"succeeded", "failed"}:
                return current
            if current.state == "cancelled":
                try:
                    reconciled = self.kube.wait_for_job_quiescence(
                        self.settings.namespace, evaluation, 10.0
                    )
                except TimeoutError as exc:
                    raise EvaluationConflict(str(exc)) from exc
                result = self._status(reconciled)
                if result.state == "cancelled":
                    self._synthesize_terminal_report(
                        reconciled, "cancelled", "cancelled by operator"
                    )
                result = self._status(reconciled)
                self._register_terminal_report(result)
                return result
            report_path = self.artifacts.path(evaluation, "report.json")
            if report_path.exists():
                EvaluationReport.model_validate(
                    self.artifacts.read_json(report_path, MAX_REPORT_BYTES)
                )
                raise EvaluationConflict(
                    "a terminal worker report exists while Kubernetes status settles"
                )
            metadata = job.get("metadata") or {}
            resource_version = _field(metadata, "resource_version", "resourceVersion")
            if not resource_version:
                raise EvaluationConflict("Kubernetes Job has no resourceVersion")
            try:
                self.kube.cancel_job(
                    self.settings.namespace,
                    evaluation,
                    _now(),
                    str(resource_version),
                )
            except Exception as exc:
                if getattr(exc, "status", None) == 409:
                    continue
                raise
            try:
                reconciled = self.kube.wait_for_job_quiescence(
                    self.settings.namespace, evaluation, 10.0
                )
            except TimeoutError as exc:
                raise EvaluationConflict(str(exc)) from exc
            result = self._status(reconciled)
            if result.state == "cancelled":
                self._synthesize_terminal_report(
                    reconciled, "cancelled", "cancelled by operator"
                )
            result = self._status(reconciled)
            self._register_terminal_report(result)
            return result
        raise EvaluationConflict("evaluation changed concurrently; retry cancellation")

    def list_artifacts(self, evaluation: str) -> ArtifactList:
        self.get(evaluation)
        return self.artifacts.list(evaluation)

    def get_report(self, evaluation: str) -> EvaluationReport:
        status = self.get(evaluation)
        path = self.artifacts.path(evaluation, "report.json")
        if status.state not in {"succeeded", "failed", "cancelled"}:
            raise EvaluationConflict("evaluation is not terminal")
        if not path.exists():
            if status.state in {"failed", "cancelled"}:
                job = self.kube.get_job(self.settings.namespace, evaluation)
                if status.state == "cancelled":
                    try:
                        job = self.kube.wait_for_job_quiescence(
                            self.settings.namespace, evaluation, 10.0
                        )
                    except TimeoutError as exc:
                        raise EvaluationConflict(str(exc)) from exc
                self._synthesize_terminal_report(
                    job,
                    status.state,
                    status.message or f"Kubernetes Job {status.state}",
                )
            else:
                raise EvaluationNotFound(f"report for {evaluation}")
        report = EvaluationReport.model_validate(
            self.artifacts.read_json(path, MAX_REPORT_BYTES)
        )
        if report.terminal_state != status.state:
            raise EvaluationConflict(
                "durable report terminal state does not match the Kubernetes Job"
            )
        return report

    def reconcile_ledger(self) -> None:
        if self.ledger is None:
            return
        jobs = self.kube.list_jobs(self.settings.namespace, self.managed_selector, None)
        for job in jobs:
            self._register_terminal_report(self._status(job))

    def _register_terminal_report(self, status: EvaluationStatus) -> None:
        if (
            self.ledger is None
            or status.state not in {"succeeded", "failed", "cancelled"}
            or status.workstream is None
        ):
            return
        output_dir = self.artifacts.output_dir(status.evaluation_id)
        report_path = output_dir / "report.json"
        marker_path = output_dir / "ledger-registration.json"
        if not report_path.exists() or marker_path.exists():
            return
        report_metadata = artifact_metadata(report_path)
        run_id = status.workstream.run_id
        artifact = {
            "artifact_id": report_metadata["sha256"],
            "kind": "lm-evaluation-report",
            "workstream_id": status.workstream.workstream_id,
            "operation_ref": status.evaluation_id,
            "storage_url": (
                f"pvc://{self.settings.namespace}/{self.settings.result_claim_name}/"
                f"{status.evaluation_id}/report.json"
            ),
            "note": f"run_id={run_id}" if run_id else None,
        }
        try:
            self.ledger.register(artifact)
            atomic_json(marker_path, artifact)
        except Exception as exc:  # noqa: BLE001 - ledger is loss-tolerant memory.
            logger.warning(
                "ledger registration failed for %s: %s", status.evaluation_id, exc
            )

    def logs(self, evaluation: str, tail_lines: int) -> str:
        self.get(evaluation)
        try:
            logs = self.kube.read_logs(self.settings.namespace, evaluation, tail_lines)
        except Exception:  # noqa: BLE001 - durable logs survive Pod/API loss.
            logs = ""
        if logs:
            return logs[-65_536:]
        return self.artifacts.tail_log(evaluation, tail_lines)

    def ready(self) -> bool:
        return self.kube.ready(self.settings.namespace) and self.artifacts.ready()

    def _synthesize_terminal_report(
        self, job: dict[str, Any], state: str, error: str
    ) -> None:
        """Persist a bounded report when Kubernetes terminates before worker output."""
        status = self._status(job)
        output_dir = self.artifacts.output_dir(status.evaluation_id)
        output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
        report_path = output_dir / "report.json"
        if report_path.exists():
            self.artifacts.ensure_manifest(output_dir, status.evaluation_id)
            return
        annotations = self._annotations(job)
        effective = json.loads(
            annotations[f"{ANNOTATION_PREFIX}effective-configuration"]
        )
        config_path = output_dir / "effective-configuration.json"
        config_path.write_text(
            json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8"
        )
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        completed = (
            annotations.get(f"{ANNOTATION_PREFIX}cancelled-at")
            if state == "cancelled"
            else status.completed_at
        )
        started = status.started_at
        latency = None
        if started is not None and completed is not None:
            latency = max(
                0.0,
                (
                    datetime.fromisoformat(str(completed))
                    - datetime.fromisoformat(str(started))
                ).total_seconds(),
            )
        report = EvaluationReport(
            evaluation_id=status.evaluation_id,
            terminal_state=state,
            metrics={},
            primary_metric=None,
            sample_counts={},
            dataset_fingerprints={},
            latency_seconds=latency,
            lm_eval_version=effective["lm_eval_version"],
            lm_eval_commit=self.lm_eval_revision,
            effective_configuration=effective,
            model_identity={
                "served_model": status.model,
                "endpoint": status.endpoint_identity,
            },
            measurement_started_at=started,
            measurement_completed_at=completed,
            artifact_hashes={config_path.name: config_hash},
            error=error[:4096],
        )
        for name in ("error.txt", "lm-eval-result.json", "samples.jsonl", "worker.log"):
            candidate = output_dir / name
            if candidate.exists():
                metadata = artifact_metadata(candidate)
                report.artifact_hashes[name] = metadata["sha256"]
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir, prefix=".report-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report.model_dump(mode="json"), stream, indent=2, sort_keys=True)
        try:
            os.link(temporary, report_path)
        except FileExistsError:
            # A worker reached a durable terminal result concurrently. It owns
            # that provenance; reconciliation must never replace it.
            return
        finally:
            temporary.unlink(missing_ok=True)
        self.artifacts.ensure_manifest(output_dir, status.evaluation_id)

    @staticmethod
    def _annotations(job: dict[str, Any]) -> dict[str, str]:
        return _field(job.get("metadata", {}), "annotations") or {}

    def _status(self, job: dict[str, Any]) -> EvaluationStatus:
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
        if state == "pending" and f"{ANNOTATION_PREFIX}cancelled-at" in annotations:
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
                    durable = EvaluationReport.model_validate(
                        self.artifacts.read_json(report_path, MAX_REPORT_BYTES)
                    )
                except (OSError, ValueError, EvaluationConflict, ArtifactConflict):
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
        workstream = request.get("workstream")
        return EvaluationStatus(
            evaluation_id=metadata["name"],
            state=state,
            created_at=_field(metadata, "creation_timestamp", "creationTimestamp"),
            started_at=_field(status, "start_time", "startTime"),
            completed_at=completed_at,
            endpoint_identity=annotations[f"{ANNOTATION_PREFIX}endpoint-identity"],
            model=request["model"],
            profile=request["profile"],
            request_sha256=annotations[f"{ANNOTATION_PREFIX}request-sha256"],
            workstream=workstream,
            message=message,
        )
