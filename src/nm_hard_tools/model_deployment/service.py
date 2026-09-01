"""Idempotent orchestration for the single deployment operation."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.models import DeployModelResult
from nm_hard_tools.model_deployment.renderer import (
    ManifestoConfigError,
    ManifestoRenderer,
    OperatorConfigurationError,
)
from nm_hard_tools.model_deployment.runtime import DeploymentRuntime, RuntimeFailure


class DeployFailure(Exception):
    def __init__(
        self, code: str, message: str, retryable: bool, deployment_id: str | None = None
    ) -> None:
        super().__init__(message[:1024])
        self.code = code
        self.retryable = retryable
        self.deployment_id = deployment_id


class DeploymentService:
    def __init__(
        self,
        settings: DeploymentSettings,
        runtime: DeploymentRuntime,
        renderer: ManifestoRenderer | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.renderer = renderer or ManifestoRenderer(settings)
        self._locks: defaultdict[str, Lock] = defaultdict(Lock)
        self._locks_guard = Lock()

    def deploy(self, manifesto_config: str) -> DeployModelResult:
        try:
            rendered = self.renderer.render(manifesto_config)
        except ManifestoConfigError as exc:
            raise DeployFailure("INVALID_MANIFESTO_CONFIG", str(exc), False) from exc
        except OperatorConfigurationError as exc:
            raise DeployFailure(
                "INVALID_OPERATOR_CONFIGURATION",
                "Operator render configuration is invalid",
                False,
            ) from exc

        with self._locks_guard:
            deployment_lock = self._locks[rendered.deployment_id]
        with deployment_lock:
            try:
                self.runtime.deploy(rendered, self.settings.readiness_timeout_seconds)
            except RuntimeFailure as exc:
                deployment_id = rendered.deployment_id if exc.accepted else None
                if exc.code == "READINESS_TIMEOUT":
                    deployment_id = rendered.deployment_id
                raise DeployFailure(
                    exc.code, exc.message, exc.retryable, deployment_id
                ) from exc
        return DeployModelResult(
            manifesto_digest=rendered.manifesto_digest,
            deployment_id=rendered.deployment_id,
            endpoint=rendered.endpoint,
            target_namespace=rendered.namespace,
            resources=rendered.resources,
        )
