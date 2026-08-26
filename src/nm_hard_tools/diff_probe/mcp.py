"""Probe tools for the shared stateless MCP transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from nm_hard_tools.diff_probe.models import (
    ListProbesRequest,
    ProbeArtifactList,
    ProbeList,
    ProbePlanResponse,
    ProbeRef,
    ProbeReport,
    ProbeRequest,
    ProbeStatus,
)
from nm_hard_tools.mcp import McpTool, McpToolResult

if TYPE_CHECKING:
    from nm_hard_tools.diff_probe.service import ProbeService


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def probe_tools(service: ProbeService) -> list[McpTool]:
    """Service-specific tools with shared protocol handling."""
    read = {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }

    def handler(
        model: type[BaseModel],
        callback: Callable[[Any], BaseModel],
    ) -> Callable[[dict[str, Any]], McpToolResult]:
        def invoke(arguments: dict[str, Any]) -> McpToolResult:
            try:
                value = callback(model.model_validate(arguments))
            except (ValidationError, ValueError, KeyError, RuntimeError) as exc:
                return McpToolResult(text=str(exc)[:4096], is_error=True)
            structured = value.model_dump(mode="json")
            return McpToolResult(
                text=json.dumps(structured, sort_keys=True, separators=(",", ":")),
                structured_content=structured,
            )

        return invoke

    return [
        McpTool(
            name="plan_probe",
            description=(
                "Validate a differential or smoke probe against the operator "
                "allowlists and render its exact Job without creating it."
            ),
            input_schema=_schema(ProbeRequest),
            output_schema=_schema(ProbePlanResponse),
            annotations=read,
            handler=handler(ProbeRequest, service.plan),
        ),
        McpTool(
            name="submit_probe",
            description="Idempotently create the planned CPU-only Kubernetes Job.",
            input_schema=_schema(ProbeRequest),
            output_schema=_schema(ProbeStatus),
            annotations={**read, "readOnlyHint": False},
            handler=handler(ProbeRequest, service.submit),
        ),
        McpTool(
            name="list_probes",
            description="List bounded probe summaries reconstructed from Jobs.",
            input_schema=_schema(ListProbesRequest),
            output_schema=_schema(ProbeList),
            annotations=read,
            handler=handler(ListProbesRequest, lambda value: service.list(value.limit)),
        ),
        McpTool(
            name="get_probe",
            description="Get one probe summary from its Kubernetes Job.",
            input_schema=_schema(ProbeRef),
            output_schema=_schema(ProbeStatus),
            annotations=read,
            handler=handler(ProbeRef, lambda value: service.get(value.probe_id)),
        ),
        McpTool(
            name="cancel_probe",
            description=(
                "Idempotently suspend an active probe Job and retain its record."
            ),
            input_schema=_schema(ProbeRef),
            output_schema=_schema(ProbeStatus),
            annotations={**read, "readOnlyHint": False, "destructiveHint": True},
            handler=handler(ProbeRef, lambda value: service.cancel(value.probe_id)),
        ),
        McpTool(
            name="get_probe_report",
            description=(
                "Get the durable terminal report: bounded metrics, fingerprints, "
                "and content hashes; sample bodies remain artifacts."
            ),
            input_schema=_schema(ProbeRef),
            output_schema=_schema(ProbeReport),
            annotations=read,
            handler=handler(ProbeRef, lambda value: service.get_report(value.probe_id)),
        ),
        McpTool(
            name="list_probe_artifacts",
            description="List bounded artifact metadata; bodies are not returned.",
            input_schema=_schema(ProbeRef),
            output_schema=_schema(ProbeArtifactList),
            annotations=read,
            handler=handler(
                ProbeRef, lambda value: service.list_artifacts(value.probe_id)
            ),
        ),
    ]
