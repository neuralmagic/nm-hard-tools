"""Strict public schemas for the differential probe service."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

StrictName = Annotated[
    str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,62}$", max_length=63)
]
ProbeId = Annotated[
    str, StringConstraints(pattern=r"^probe-[a-f0-9]{20}$", max_length=26)
]


class StrictModel(BaseModel):
    """Base for API types that reject all undeclared input."""

    model_config = ConfigDict(extra="forbid")


class ProbeRequest(StrictModel):
    mode: Literal["diff", "smoke"]
    target_a: StrictName
    target_b: StrictName | None = None
    model: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    request_set: StrictName
    limit: int | None = Field(default=None, ge=1, le=512)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    timeout_seconds: int = Field(default=3_600, ge=60, le=21_600)

    @model_validator(mode="after")
    def coherent_targets(self) -> ProbeRequest:
        if self.mode == "diff":
            if self.target_b is None:
                raise ValueError("diff mode requires target_b")
            if self.target_b == self.target_a:
                raise ValueError(
                    "diff mode compares two distinct targets; target_a and "
                    "target_b are the same"
                )
        elif self.target_b is not None:
            raise ValueError("smoke mode takes a single target")
        return self


class ProbeRef(StrictModel):
    probe_id: ProbeId


class ListProbesRequest(StrictModel):
    limit: int = Field(default=50, ge=1, le=100)


class ProbePlanResponse(StrictModel):
    probe_id: ProbeId
    request_sha256: str
    plan_sha256: str
    endpoint_identities: dict[str, str]
    effective_configuration: dict[str, Any]
    resources: list[dict[str, Any]]


class ProbeStatus(StrictModel):
    probe_id: ProbeId
    state: Literal["planned", "pending", "running", "succeeded", "failed", "cancelled"]
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    mode: Literal["diff", "smoke"]
    model: str
    request_set: StrictName
    endpoint_identities: dict[str, str]
    request_sha256: str
    message: str | None = None


class ProbeList(StrictModel):
    probes: list[ProbeStatus]


class ProbeArtifactMetadata(StrictModel):
    name: Annotated[
        str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    ]
    media_type: Annotated[str, StringConstraints(max_length=128)]
    size_bytes: int = Field(ge=0)
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class ProbeArtifactList(StrictModel):
    probe_id: ProbeId
    artifacts: list[ProbeArtifactMetadata]


class ProbeReport(StrictModel):
    probe_id: ProbeId
    terminal_state: Literal["succeeded", "failed", "cancelled"]
    mode: Literal["diff", "smoke"]
    # Bounded finite numeric readings; the primary one is named separately so a
    # caller never guesses. diff: mean_divergence; smoke: pass_rate.
    metrics: dict[str, float]
    primary_metric: str
    request_count: int = Field(ge=0)
    request_set_fingerprint: dict[str, str]
    endpoint_identities: dict[str, str]
    model_identity: dict[str, str]
    effective_configuration: dict[str, Any]
    measurement_started_at: datetime | None = None
    measurement_completed_at: datetime | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    artifact_hashes: dict[str, str]
    service_version: str
    error: str | None = None


def canonical_request(request: ProbeRequest) -> str:
    """The canonical, bounded representation used for identity."""
    return json.dumps(
        request.model_dump(mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )


def request_sha256(request: ProbeRequest) -> str:
    return hashlib.sha256(canonical_request(request).encode()).hexdigest()


def probe_id_from_plan(plan_sha256: str) -> str:
    return f"probe-{plan_sha256[:20]}"


def validate_probe_id(value: str) -> str:
    if not re.fullmatch(r"probe-[a-f0-9]{20}", value):
        raise ValueError("probe_id must match probe-<20 hex characters>")
    return value
