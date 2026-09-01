"""Kubernetes create-only boundary for rendered Manifesto deployments."""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import yaml

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.renderer import (
    DIGEST_ANNOTATION,
    OWNER_ANNOTATION,
    RenderedDeployment,
)


@dataclass(frozen=True)
class RuntimeFailure(Exception):
    code: str
    message: str
    retryable: bool
    accepted: bool = False

    def __str__(self) -> str:
        return self.message


class DeploymentRuntime(Protocol):
    def deploy(self, rendered: RenderedDeployment, timeout_seconds: int) -> None: ...


class KubectlRuntime:
    def __init__(self, settings: DeploymentSettings) -> None:
        self.settings = settings

    def _base(self) -> list[str]:
        command = ["kubectl"]
        if self.settings.kube_context:
            command.extend(["--context", self.settings.kube_context])
        command.extend(["--namespace", self.settings.namespace])
        return command

    def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [*self._base(), *args],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout[-65_536:],
                stderr=result.stderr[-65_536:],
            )
        return result

    def deploy(self, rendered: RenderedDeployment, timeout_seconds: int) -> None:
        absent: list[dict[str, Any]] = []
        try:
            for obj in rendered.objects:
                existing = self._get(obj)
                if existing is None:
                    absent.append(obj)
                else:
                    self._validate_existing(obj, existing, rendered.deployment_id)
        except RuntimeFailure:
            raise
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise RuntimeFailure(
                "DEPENDENCY_UNAVAILABLE",
                "Kubernetes discovery is unavailable",
                True,
            ) from exc

        if absent:
            dry_run_yaml = yaml.safe_dump_all(
                absent, sort_keys=False, explicit_start=True
            )
            try:
                self._run(
                    ["create", "--dry-run=server", "-f", "-"], input_text=dry_run_yaml
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeFailure(
                    "TARGET_VALIDATION_FAILED",
                    "Rendered resources failed Kubernetes server validation",
                    False,
                ) from exc

        accepted = False
        for obj in absent:
            body = yaml.safe_dump(obj, sort_keys=False)
            try:
                result = self._run(["create", "-f", "-"], input_text=body, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeFailure(
                    "DEPENDENCY_UNAVAILABLE",
                    "Kubernetes creation is unavailable",
                    True,
                    accepted,
                ) from exc
            if result.returncode == 0:
                accepted = True
                continue
            try:
                existing = self._get(obj)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise RuntimeFailure(
                    "DEPENDENCY_UNAVAILABLE",
                    "Kubernetes creation outcome is unavailable",
                    True,
                    accepted,
                ) from exc
            if existing is not None:
                try:
                    self._validate_existing(obj, existing, rendered.deployment_id)
                except RuntimeFailure as exc:
                    raise RuntimeFailure(
                        exc.code,
                        exc.message,
                        exc.retryable,
                        accepted,
                    ) from exc
                continue
            raise RuntimeFailure(
                "APPLY_FAILED",
                "Kubernetes rejected a rendered resource",
                True,
                accepted,
            )

        self._wait_ready(rendered, timeout_seconds, accepted)

    def _get(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        body = yaml.safe_dump(obj, sort_keys=False)
        result = self._run(
            ["get", "-f", "-", "--ignore-not-found=true", "-o", "json"],
            input_text=body,
            check=False,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        if not result.stdout.strip():
            return None
        value = json.loads(result.stdout)
        if value.get("kind") == "List":
            items = value.get("items", [])
            return items[0] if items else None
        return value

    @staticmethod
    def _validate_existing(
        expected: dict[str, Any], existing: dict[str, Any], deployment_id: str
    ) -> None:
        expected_annotations = expected.get("metadata", {}).get("annotations", {})
        actual_annotations = existing.get("metadata", {}).get("annotations", {})
        if actual_annotations.get(
            OWNER_ANNOTATION
        ) != deployment_id or actual_annotations.get(
            DIGEST_ANNOTATION
        ) != expected_annotations.get(DIGEST_ANNOTATION):
            raise RuntimeFailure(
                "APPLY_FAILED",
                "A rendered resource already exists with different ownership or intent",
                False,
            )

    def _wait_ready(
        self, rendered: RenderedDeployment, timeout_seconds: int, accepted: bool
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                workloads_ready = all(
                    self._workload_ready(
                        workload.pod_selector,
                        workload.expected_pods,
                        rendered.deployment_id,
                        rendered.intent_digest,
                        deadline,
                    )
                    for workload in rendered.workloads
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise RuntimeFailure(
                    "DEPENDENCY_UNAVAILABLE",
                    "Kubernetes readiness observation is unavailable",
                    True,
                    accepted,
                ) from exc
            if (
                workloads_ready
                and (remaining := deadline - time.monotonic()) > 0
                and self._endpoint_ready(rendered.endpoint, remaining)
                and time.monotonic() <= deadline
            ):
                return
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        raise RuntimeFailure(
            "READINESS_TIMEOUT",
            "Model workloads or inference endpoint did not become ready in time",
            True,
            accepted,
        )

    def _workload_ready(
        self,
        pod_selector: dict[str, str],
        expected_pods: int,
        deployment_id: str,
        manifesto_digest: str,
        deadline: float,
    ) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        selector = ",".join(
            f"{key}={value}" for key, value in sorted(pod_selector.items())
        )
        result = self._run(
            ["get", "pods", "-l", selector, "-o", "json"],
            timeout=max(1, min(30, math.ceil(remaining))),
        )
        pods = json.loads(result.stdout).get("items", [])
        return len(pods) >= expected_pods and all(
            _pod_ready(pod, deployment_id, manifesto_digest) for pod in pods
        )

    @staticmethod
    def _endpoint_ready(endpoint: str, timeout_seconds: float) -> bool:
        try:
            response = httpx.get(
                endpoint,
                timeout=max(0.001, min(10, timeout_seconds)),
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return False
        if not 200 <= response.status_code < 300:
            return False
        try:
            value = response.json()
        except ValueError:
            return False
        return isinstance(value, dict) and isinstance(value.get("data"), list)


def _pod_ready(pod: dict[str, Any], deployment_id: str, manifesto_digest: str) -> bool:
    annotations = pod.get("metadata", {}).get("annotations", {})
    if (
        annotations.get(OWNER_ANNOTATION) != deployment_id
        or annotations.get(DIGEST_ANNOTATION) != manifesto_digest
    ):
        return False
    conditions = pod.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )
