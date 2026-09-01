"""Closed MCP schemas for the model deployment tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeployModelInput(StrictModel):
    manifesto_config: str = Field(
        min_length=1,
        max_length=1_048_576,
        description="One complete llm-manifesto model configuration encoded as YAML.",
    )

    @field_validator("manifesto_config")
    @classmethod
    def bounded_utf8(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 1_048_576:
            raise ValueError("manifesto_config exceeds the 1 MiB UTF-8 limit")
        return value


class ResourceRef(StrictModel):
    api_version: str = Field(min_length=1, max_length=253)
    kind: str = Field(min_length=1, max_length=253)
    name: str = Field(min_length=1, max_length=253)


class DeployModelResult(StrictModel):
    schema_version: Literal["1"] = "1"
    manifesto_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    deployment_id: str = Field(min_length=1, max_length=253)
    endpoint: str = Field(min_length=1, max_length=2048)
    ready: Literal[True] = True
    target_namespace: str = Field(min_length=1, max_length=63)
    resources: list[ResourceRef] = Field(max_length=64)


ErrorCode = Literal[
    "INVALID_MANIFESTO_CONFIG",
    "INVALID_OPERATOR_CONFIGURATION",
    "TARGET_VALIDATION_FAILED",
    "APPLY_FAILED",
    "READINESS_TIMEOUT",
    "DEPENDENCY_UNAVAILABLE",
]


class DeployModelError(StrictModel):
    schema_version: Literal["1"] = "1"
    code: ErrorCode
    message: str = Field(min_length=1, max_length=1024)
    retryable: bool
    deployment_id: str | None = Field(default=None, min_length=1, max_length=253)


def output_schema() -> dict:
    schema = TypeAdapter(DeployModelResult | DeployModelError).json_schema()
    schema["oneOf"] = schema.pop("anyOf")
    return schema
