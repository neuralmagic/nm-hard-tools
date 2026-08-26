"""Operator configuration: target allowlist and the frozen request-set catalog."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nm_hard_tools.diff_probe.models import StrictName
from nm_hard_tools.evaluation_service.config import TargetConfig

__all__ = ["ProbeServiceSettings", "RequestSetConfig", "TargetConfig"]


class RequestSetConfig(BaseModel):
    """One frozen request set: a JSONL file the operator mounted, pinned by
    content hash. The worker refuses a file whose hash moved — a request set is
    part of the measurement's identity, never a mutable input."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\.jsonl$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    api: str = Field(default="chat", pattern=r"^(chat|completions)$")
    max_requests: int = Field(default=128, ge=1, le=512)
    default_max_tokens: int = Field(default=512, ge=1, le=4096)

    @field_validator("file")
    @classmethod
    def contained_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("file must be a normalized path below the catalog root")
        return value


class ProbeServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: StrictName = "diff-probe"
    worker_image: str
    result_claim_name: StrictName
    result_root: str = "/var/lib/diff-probe-results"
    request_set_claim_name: StrictName | None = None
    request_set_root: str = "/var/lib/diff-probe-request-sets"
    worker_service_account: StrictName = "diff-probe-worker"
    service_instance: StrictName = "diff-probe-service"
    image_pull_secrets: list[StrictName] = Field(default_factory=list, max_length=16)
    node_selector: dict[str, str] = Field(default_factory=dict, max_length=32)
    cpu_request: str = "500m"
    cpu_limit: str = "2"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    worker_cache_size_limit: str = "1Gi"
    mutation_rate_limit_per_minute: int = Field(default=20, ge=1, le=600)
    allowed_origins: list[str] = Field(default_factory=list, max_length=32)
    targets: dict[StrictName, TargetConfig]
    request_sets: dict[StrictName, RequestSetConfig]

    @field_validator("worker_image")
    @classmethod
    def pinned_image(cls, value: str) -> str:
        if not re.search(r"@sha256:[a-f0-9]{64}$", value):
            raise ValueError("worker_image must be pinned by sha256 digest")
        return value

    @field_validator("result_root", "request_set_root")
    @classmethod
    def absolute_normalized(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized or not value.startswith("/") or ".." in Path(value).parts:
            raise ValueError("roots must be absolute normalized paths")
        return normalized

    @classmethod
    def from_file(cls, path: str | Path) -> ProbeServiceSettings:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise TypeError("settings document must be an object")
        return cls.model_validate(data)

    @classmethod
    def from_environment(cls) -> ProbeServiceSettings:
        path = os.environ.get("DIFF_PROBE_SERVICE_CONFIG")
        if path:
            return cls.from_file(path)
        raw = os.environ.get("DIFF_PROBE_SERVICE_CONFIG_JSON")
        if not raw:
            raise RuntimeError(
                "DIFF_PROBE_SERVICE_CONFIG or DIFF_PROBE_SERVICE_CONFIG_JSON "
                "is required"
            )
        return cls.model_validate(json.loads(raw))
