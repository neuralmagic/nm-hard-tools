"""Evaluation tools for the shared stateless MCP transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from inference_hard_tools.evaluation_service.models import (
    ArtifactList,
    EvaluationList,
    EvaluationRef,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    ListEvaluationsRequest,
    PlanResponse,
)
from inference_hard_tools.mcp import McpTool, McpToolResult

if TYPE_CHECKING:
    from inference_hard_tools.evaluation_service.service import EvaluationService


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def evaluation_tools(
    service: EvaluationService,
) -> list[McpTool]:
    """Return service-specific tools with shared protocol handling."""
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
            name="plan_evaluation",
            description=(
                "Validate an operator-approved evaluation and render its exact Job "
                "without creating it."
            ),
            input_schema=_schema(EvaluationRequest),
            output_schema=_schema(PlanResponse),
            annotations=read,
            handler=handler(EvaluationRequest, service.plan),
        ),
        McpTool(
            name="submit_evaluation",
            description="Idempotently create the planned CPU-only Kubernetes Job.",
            input_schema=_schema(EvaluationRequest),
            output_schema=_schema(EvaluationStatus),
            annotations={**read, "readOnlyHint": False},
            handler=handler(EvaluationRequest, service.submit),
        ),
        McpTool(
            name="list_evaluations",
            description=(
                "List bounded evaluation summaries reconstructed from Kubernetes Jobs."
            ),
            input_schema=_schema(ListEvaluationsRequest),
            output_schema=_schema(EvaluationList),
            annotations=read,
            handler=handler(
                ListEvaluationsRequest, lambda value: service.list(value.limit)
            ),
        ),
        McpTool(
            name="get_evaluation",
            description="Get one evaluation summary from its Kubernetes Job.",
            input_schema=_schema(EvaluationRef),
            output_schema=_schema(EvaluationStatus),
            annotations=read,
            handler=handler(
                EvaluationRef, lambda value: service.get(value.evaluation_id)
            ),
        ),
        McpTool(
            name="cancel_evaluation",
            description=(
                "Idempotently suspend an active evaluation Job and retain its "
                "operation record."
            ),
            input_schema=_schema(EvaluationRef),
            output_schema=_schema(EvaluationStatus),
            annotations={**read, "readOnlyHint": False, "destructiveHint": True},
            handler=handler(
                EvaluationRef,
                lambda value: service.cancel(value.evaluation_id),
            ),
        ),
        McpTool(
            name="list_evaluation_artifacts",
            description=(
                "List bounded artifact metadata; artifact bodies are not returned."
            ),
            input_schema=_schema(EvaluationRef),
            output_schema=_schema(ArtifactList),
            annotations=read,
            handler=handler(
                EvaluationRef,
                lambda value: service.list_artifacts(value.evaluation_id),
            ),
        ),
        McpTool(
            name="get_evaluation_report",
            description=(
                "Get the bounded terminal report; samples and logs remain artifacts."
            ),
            input_schema=_schema(EvaluationRef),
            output_schema=_schema(EvaluationReport),
            annotations=read,
            handler=handler(
                EvaluationRef, lambda value: service.get_report(value.evaluation_id)
            ),
        ),
    ]
