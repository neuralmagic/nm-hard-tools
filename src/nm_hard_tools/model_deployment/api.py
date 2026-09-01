"""Authenticated stateless MCP application for model deployment."""

from __future__ import annotations

import secrets
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from nm_hard_tools.mcp import McpServer
from nm_hard_tools.model_deployment import SERVICE_VERSION
from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.mcp import deployment_tools
from nm_hard_tools.model_deployment.renderer import ManifestoRenderer
from nm_hard_tools.model_deployment.runtime import DeploymentRuntime, KubectlRuntime
from nm_hard_tools.model_deployment.service import DeploymentService

# JSON escaping can expand a valid 1 MiB YAML string well beyond its decoded size.
MAX_REQUEST_BYTES = 8_388_608


class AuthenticatedBodyLimitMiddleware:
    """Authorize before parsing and keep the complete MCP envelope bounded."""

    def __init__(self, app: Any, token: str) -> None:
        if not token:
            raise ValueError("a non-empty bearer token is required")
        self.app = app
        self.token = token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/mcp":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")
        supplied = (
            authorization.removeprefix("Bearer ")
            if authorization.startswith("Bearer ")
            else ""
        )
        if not supplied or not secrets.compare_digest(supplied, self.token):
            response = JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32001,
                        "message": "valid bearer authorization is required",
                    },
                },
            )
            await response(scope, receive, send)
            return
        content_length = headers.get("content-length")
        try:
            declared = int(content_length) if content_length is not None else 0
        except ValueError:
            declared = MAX_REQUEST_BYTES + 1
        if declared > MAX_REQUEST_BYTES:
            await self._too_large(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                await self._too_large(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope: Any, receive: Any, send: Any) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Request too large"},
            },
        )
        await response(scope, receive, send)


def create_app(
    settings: DeploymentSettings,
    runtime: DeploymentRuntime,
    *,
    renderer: ManifestoRenderer | None = None,
    bearer_token: str | None = None,
) -> FastAPI:
    service = DeploymentService(settings, runtime, renderer)
    app = FastAPI(
        title="Manifesto model deployment MCP service",
        version=SERVICE_VERSION,
        docs_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        AuthenticatedBodyLimitMiddleware,
        token=bearer_token if bearer_token is not None else settings.token(),
    )

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    McpServer(
        name="manifesto-model-deployment",
        version=SERVICE_VERSION,
        instructions=(
            "Call deploy_model with one complete Manifesto model YAML document. "
            "The operator fixes cluster, namespace, credentials, renderer, and policy. "
            "A successful result identifies the ready in-cluster inference endpoint; "
            "rendered manifests and logs are intentionally omitted."
        ),
        tools=deployment_tools(service),
        allowed_origins=set(settings.allowed_origins),
    ).install(app)
    return app


def main() -> None:
    settings = DeploymentSettings.from_environment()
    app = create_app(settings, KubectlRuntime(settings))
    uvicorn.run(app, host="0.0.0.0", port=8000)
