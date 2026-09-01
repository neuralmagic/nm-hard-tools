"""MCP definition for the single model deployment tool."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from nm_hard_tools.mcp import McpTool, McpToolResult
from nm_hard_tools.model_deployment.models import (
    DeployModelError,
    DeployModelInput,
    FieldIssue,
    output_schema,
)
from nm_hard_tools.model_deployment.service import DeployFailure, DeploymentService


def deployment_tools(service: DeploymentService) -> list[McpTool]:
    async def deploy(arguments: dict) -> McpToolResult:
        try:
            request = DeployModelInput.model_validate(arguments)
        except ValidationError:
            error = DeployModelError(
                schema_version="1",
                code="INVALID_MANIFESTO_CONFIG",
                message="manifesto_config must be the only input and fit within 1 MiB",
                retryable=False,
                retry_after_ms=None,
                field_issues=[
                    FieldIssue(
                        field="manifesto_config",
                        code="INVALID_ARGUMENT",
                        message="Provide only one Manifesto YAML string up to 1 MiB",
                    )
                ],
                current_state=None,
                suggested_action=None,
                deployment_id=None,
            )
            return McpToolResult(
                text=error.message,
                structured_content=error.model_dump(mode="json"),
                is_error=True,
            )
        try:
            result = await asyncio.to_thread(service.deploy, request.manifesto_config)
        except DeployFailure as exc:
            field_issues = []
            if exc.code == "INVALID_MANIFESTO_CONFIG":
                field_issues = [
                    FieldIssue(
                        field=exc.field or "manifesto_config",
                        code="INVALID_ARGUMENT",
                        message=str(exc),
                    )
                ]
            error = DeployModelError(
                schema_version="1",
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
                retry_after_ms=1_000 if exc.retryable else None,
                field_issues=field_issues,
                current_state=None,
                suggested_action=None,
                deployment_id=exc.deployment_id,
            )
            return McpToolResult(
                text=error.message,
                structured_content=error.model_dump(mode="json"),
                is_error=True,
            )
        structured = result.model_dump(mode="json")
        return McpToolResult(
            text=f"Deployment {result.deployment_id} is ready at {result.endpoint}",
            structured_content=structured,
        )

    return [
        McpTool(
            name="deploy_model",
            title="Deploy model",
            description=(
                "Deploy one caller-supplied Manifesto model configuration into the "
                "operator-controlled target; creates resources and returns their ready "
                "inference endpoint after authorization and validation."
            ),
            input_schema=DeployModelInput.model_json_schema(),
            output_schema=output_schema(),
            annotations={
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=deploy,
        )
    ]
