"""Operator configuration with target allowlisting and immutable worker inputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_hard_tools.evaluation_service.models import SecretKey, StrictName

HF_TOKEN_SECRET_KEY = "HF_TOKEN"


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    auth_secret_name: StrictName | None = None
    auth_secret_key: SecretKey = "api-key"
    tls_verify: bool = True
    max_concurrency: int = Field(default=8, ge=1, le=64)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, query, or fragment")
        return value.rstrip("/")

    def endpoint(self, chat: bool) -> str:
        suffix = "/chat/completions" if chat else "/completions"
        return f"{self.base_url}{suffix}"

    def endpoint_identity(self, chat: bool) -> str:
        parsed = urlsplit(self.endpoint(chat))
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


class DatasetConfig(BaseModel):
    """Immutable Hugging Face dataset identity for one installed task."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
    )
    name: str | None = Field(default=None, max_length=128)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")


class TaskConfig(BaseModel):
    """An installed lm-eval task plus its pinned dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    config_path: str
    dataset: DatasetConfig

    @field_validator("config_path")
    @classmethod
    def installed_task_config(cls, value: str) -> str:
        parts = Path(value).parts
        if (
            len(parts) < 4
            or parts[:2] != ("lm_eval", "tasks")
            or ".." in parts
            or not value.endswith((".yaml", ".yml"))
        ):
            raise ValueError("config_path must name an installed lm_eval/tasks YAML")
        return value


class MetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    metric: str = Field(min_length=1, max_length=128)
    filter: str = Field(min_length=1, max_length=128)


class EvaluationProfile(BaseModel):
    """Operator-owned, bounded evaluation definition."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskConfig] = Field(min_length=1, max_length=32)
    primary_metric: MetricConfig
    default_num_fewshot: int = Field(default=0, ge=0, le=32)
    max_num_fewshot: int = Field(default=32, ge=0, le=32)
    default_generation_limit: int = Field(default=512, ge=1, le=4096)
    max_generation_limit: int = Field(default=4096, ge=1, le=4096)
    max_samples: int = Field(default=100_000, ge=1, le=1_000_000)
    dataset_token_secret_name: StrictName | None = None
    dataset_token_secret_key: SecretKey | None = None

    @model_validator(mode="after")
    def coherent_bounds_and_metric(self) -> EvaluationProfile:
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("profile task names must be unique")
        if self.primary_metric.task not in names:
            raise ValueError("primary_metric.task must name a profile task")
        if self.default_num_fewshot > self.max_num_fewshot:
            raise ValueError("default_num_fewshot exceeds max_num_fewshot")
        if self.default_generation_limit > self.max_generation_limit:
            raise ValueError("default_generation_limit exceeds max_generation_limit")
        return self


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: StrictName = "lm-eval"
    worker_image: str
    result_claim_name: StrictName
    result_root: str = "/var/lib/lm-eval-results"
    worker_service_account: StrictName = "lm-eval-worker"
    service_instance: StrictName = "lm-eval-service"
    image_pull_secrets: list[StrictName] = Field(default_factory=list, max_length=16)
    node_selector: dict[str, str] = Field(default_factory=dict, max_length=32)
    cpu_request: str = "1"
    cpu_limit: str = "4"
    memory_request: str = "2Gi"
    memory_limit: str = "8Gi"
    worker_cache_size_limit: str = "4Gi"
    dataset_cache_claim_name: StrictName | None = None
    dataset_cache_root: str = "/var/cache/lm-eval"
    hf_token_secret_name: StrictName | None = None
    hf_token_secret_key: SecretKey = HF_TOKEN_SECRET_KEY
    mutation_rate_limit_per_minute: int = Field(default=20, ge=1, le=600)
    allowed_origins: list[str] = Field(default_factory=list, max_length=32)
    targets: dict[StrictName, TargetConfig]
    profiles: dict[StrictName, EvaluationProfile]

    @field_validator("worker_image")
    @classmethod
    def pinned_image(cls, value: str) -> str:
        if not re.search(r"@sha256:[a-f0-9]{64}$", value):
            raise ValueError("worker_image must be pinned by sha256 digest")
        return value

    @field_validator("result_root")
    @classmethod
    def absolute_result_root(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized or not value.startswith("/") or ".." in Path(value).parts:
            raise ValueError(
                "result_root must be an absolute normalized path below the root"
            )
        return normalized

    @field_validator("dataset_cache_root")
    @classmethod
    def absolute_dataset_cache_root(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized or not value.startswith("/") or ".." in Path(value).parts:
            raise ValueError("dataset_cache_root must be an absolute normalized path")
        return normalized

    @classmethod
    def from_file(cls, path: str | Path) -> ServiceSettings:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise TypeError("settings document must be an object")
        return cls.model_validate(data)

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        path = os.environ.get("LM_EVAL_SERVICE_CONFIG")
        if path:
            return cls.from_file(path)
        raw = os.environ.get("LM_EVAL_SERVICE_CONFIG_JSON")
        if not raw:
            raise RuntimeError(
                "LM_EVAL_SERVICE_CONFIG or LM_EVAL_SERVICE_CONFIG_JSON is required"
            )
        return cls.model_validate(json.loads(raw))
