"""Strict public schemas for operator-approved lm-eval profiles."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

StrictName = Annotated[
    str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,62}$", max_length=63)
]
SecretKey = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,253}$", max_length=253)
]
EvaluationId = Annotated[
    str, StringConstraints(pattern=r"^eval-[a-f0-9]{20}$", max_length=25)
]


class StrictModel(BaseModel):
    """Base for API types that reject all undeclared input."""

    model_config = ConfigDict(extra="forbid")


class Seeds(StrictModel):
    python: int = Field(default=0, ge=0, le=4_294_967_295)
    numpy: int = Field(default=1234, ge=0, le=4_294_967_295)
    torch: int = Field(default=1234, ge=0, le=4_294_967_295)
    fewshot: int = Field(default=1234, ge=0, le=4_294_967_295)


class EvaluationRequest(StrictModel):
    target: StrictName
    model: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    idempotency_key: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=256,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
            ),
        ]
        | None
    ) = None
    profile: StrictName = "gsm8k"
    num_fewshot: int | None = Field(default=None, ge=0, le=32)
    limit: int | None = Field(default=None, ge=1, le=1_000_000)
    seeds: Seeds = Field(default_factory=Seeds)
    generation_limit: int | None = Field(default=None, ge=1, le=4096)
    chat_template: Literal["none", "default"] = "none"
    timeout_seconds: int = Field(default=7_200, ge=60, le=86_400)


class EvaluationRef(StrictModel):
    evaluation_id: EvaluationId


class ListEvaluationsRequest(StrictModel):
    limit: int = Field(default=50, ge=1, le=100)


class PlanResponse(StrictModel):
    evaluation_id: EvaluationId
    request_sha256: str
    plan_sha256: str
    endpoint_identity: str
    effective_configuration: dict[str, Any]
    resources: list[dict[str, Any]]


class EvaluationStatus(StrictModel):
    evaluation_id: EvaluationId
    state: Literal["planned", "pending", "running", "succeeded", "failed", "cancelled"]
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    endpoint_identity: str
    model: str
    profile: StrictName
    request_sha256: str
    message: str | None = None


class EvaluationList(StrictModel):
    evaluations: list[EvaluationStatus]


class ArtifactMetadata(StrictModel):
    name: Annotated[
        str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    ]
    media_type: Annotated[str, StringConstraints(max_length=128)]
    size_bytes: int = Field(ge=0)
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class ArtifactList(StrictModel):
    evaluation_id: EvaluationId
    artifacts: list[ArtifactMetadata]


class PrimaryMetric(StrictModel):
    task: str
    metric: str
    filter: str
    value: float


class EvaluationReport(StrictModel):
    evaluation_id: EvaluationId
    terminal_state: Literal["succeeded", "failed", "cancelled"]
    metrics: dict[str, dict[str, float]]
    primary_metric: PrimaryMetric | None = None
    sample_counts: dict[str, int]
    dataset_fingerprints: dict[str, dict[str, str]]
    latency_seconds: float | None = Field(default=None, ge=0)
    lm_eval_version: str
    lm_eval_commit: str
    effective_configuration: dict[str, Any]
    model_identity: dict[str, str]
    measurement_started_at: datetime | None = None
    measurement_completed_at: datetime | None = None
    artifact_hashes: dict[str, str]
    error: str | None = None


def canonical_request(request: EvaluationRequest) -> str:
    """Return the canonical, bounded representation used for identity."""
    return json.dumps(
        request.model_dump(mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )


def request_sha256(request: EvaluationRequest) -> str:
    return hashlib.sha256(canonical_request(request).encode()).hexdigest()


def evaluation_id_from_plan(plan_sha256: str) -> str:
    return f"eval-{plan_sha256[:20]}"


def validate_evaluation_id(value: str) -> str:
    if not re.fullmatch(r"eval-[a-f0-9]{20}", value):
        raise ValueError("invalid evaluation_id")
    return value
