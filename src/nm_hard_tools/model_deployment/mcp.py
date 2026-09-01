"""MCP definition for the single model deployment tool."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from nm_hard_tools.mcp import McpTool, McpToolResult
from nm_hard_tools.model_deployment.models import (
    DeployModelError,
    DeployModelInput,
    output_schema,
)
from nm_hard_tools.model_deployment.service import DeployFailure, DeploymentService


def deployment_tools(service: DeploymentService) -> list[McpTool]:
    async def deploy(arguments: dict) -> McpToolResult:
        try:
            request = DeployModelInput.model_validate(arguments)
        except ValidationError:
            error = DeployModelError(
                code="INVALID_MANIFESTO_CONFIG",
                message="manifesto_config must be the only input and fit within 1 MiB",
                retryable=False,
            )
            return McpToolResult(
                text=error.message,
                structured_content=error.model_dump(mode="json"),
                is_error=True,
            )
        try:
            result = await asyncio.to_thread(service.deploy, request.manifesto_config)
        except DeployFailure as exc:
            error = DeployModelError(
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
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
                "openWorldHint": True,
            },
            handler=deploy,
        )
    ]
