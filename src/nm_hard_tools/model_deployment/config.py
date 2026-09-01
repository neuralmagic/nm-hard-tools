"""Operator-owned render and cluster configuration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeploymentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_profile: Path
    namespace: str = Field(pattern=r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
    kube_context: str | None = Field(default=None, min_length=1, max_length=253)
    bearer_token_file: Path
    readiness_timeout_seconds: int = Field(default=1800, ge=30, le=7200)
    allowed_origins: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("cluster_profile", "bearer_token_file")
    @classmethod
    def absolute_file(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("operator file paths must be absolute")
        return value

    def token(self) -> str:
        value = self.bearer_token_file.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError("bearer token file is empty")
        return value

    def render_context_identity(self, renderer_version: str) -> str:
        profile = self.cluster_profile.read_bytes()
        value = {
            "schema": 1,
            "renderer_version": renderer_version,
            "cluster_profile_sha256": hashlib.sha256(profile).hexdigest(),
            "namespace": self.namespace,
            "kube_context": self.kube_context,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_environment(cls) -> DeploymentSettings:
        raw = os.environ.get("MODEL_DEPLOY_SETTINGS_JSON")
        if raw:
            return cls.model_validate_json(raw)
        path = os.environ.get("MODEL_DEPLOY_SETTINGS")
        if not path:
            raise RuntimeError(
                "MODEL_DEPLOY_SETTINGS or MODEL_DEPLOY_SETTINGS_JSON is required"
            )
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("model deployment settings must be an object")
        return cls.model_validate(data)
