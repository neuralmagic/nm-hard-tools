"""Bounded REST and stateless Streamable HTTP MCP application."""

from __future__ import annotations

import os
import secrets
import time
from collections import deque
from threading import Lock
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from inference_hard_tools.evaluation_service import SERVICE_VERSION
from inference_hard_tools.evaluation_service.artifacts import ArtifactConflict
from inference_hard_tools.evaluation_service.config import ServiceSettings
from inference_hard_tools.evaluation_service.kube import (
    KubernetesBackend,
    KubernetesClient,
)
from inference_hard_tools.evaluation_service.mcp import evaluation_tools
from inference_hard_tools.evaluation_service.models import (
    ArtifactList,
    EvaluationList,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    PlanResponse,
)
from inference_hard_tools.evaluation_service.service import (
    EvaluationConflict,
    EvaluationNotFound,
    EvaluationService,
)
from inference_hard_tools.mcp import McpServer, McpTool

MAX_MCP_REQUEST_BYTES = 1_048_576


class MutationGuard:
    """Bearer authorization plus a process-wide bounded mutation rate."""

    def __init__(self, token: str, limit_per_minute: int) -> None:
        if not token:
            raise ValueError("a non-empty mutation bearer token is required")
        self._token = token
        self._limit = limit_per_minute
        self._attempts: deque[float] = deque()
        self._lock = Lock()

    def authorize(self, authorization: str) -> None:
        prefix = "Bearer "
        supplied = (
            authorization.removeprefix(prefix)
            if authorization.startswith(prefix)
            else ""
        )
        if not supplied or not secrets.compare_digest(supplied, self._token):
            raise HTTPException(
                status_code=401,
                detail="valid bearer authorization is required for mutations",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def check_rate(self) -> None:
        now = time.monotonic()
        with self._lock:
            while self._attempts and self._attempts[0] <= now - 60:
                self._attempts.popleft()
            if len(self._attempts) >= self._limit:
                raise HTTPException(
                    status_code=429,
                    detail="mutation rate limit exceeded",
                    headers={"Retry-After": "60"},
                )
            self._attempts.append(now)


class RequestGateMiddleware:
    """Authenticate mutations, then stream request bodies into a fixed bound."""

    def __init__(self, app: Any, guard: MutationGuard, max_body_bytes: int) -> None:
        self.app = app
        self.guard = guard
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        protected = (
            path == "/mcp"
            or (path == "/v1/evaluations" and method == "POST")
            or (path.startswith("/v1/evaluations/") and method == "DELETE")
        )
        if protected:
            try:
                self.guard.authorize(headers.get("authorization", ""))
            except HTTPException as exc:
                response = (
                    _rpc_error(
                        exc.status_code,
                        None,
                        -32001,
                        str(exc.detail),
                        headers=exc.headers,
                    )
                    if path == "/mcp"
                    else JSONResponse(
                        status_code=exc.status_code,
                        content={"detail": exc.detail},
                        headers=exc.headers,
                    )
                )
                await response(scope, receive, send)
                return
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            await self.app(scope, receive, send)
            return
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_body_bytes
            except ValueError:
                too_large = True
            if too_large:
                await self._reject_large(scope, receive, send, path)
                return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_bytes:
                await self._reject_large(scope, receive, send, path)
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
    async def _reject_large(scope: Any, receive: Any, send: Any, path: str) -> None:
        response = (
            _rpc_error(413, None, -32600, "Request too large")
            if path == "/mcp"
            else JSONResponse(status_code=413, content={"detail": "Request too large"})
        )
        await response(scope, receive, send)


def _rpc_error(
    status_code: int,
    request_id: str | int | None,
    code: int,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
    )


def create_app(
    settings: ServiceSettings,
    kube: KubernetesBackend,
    mutation_token: str,
) -> FastAPI:
    service = EvaluationService(settings, kube)
    mutation_guard = MutationGuard(
        mutation_token, settings.mutation_rate_limit_per_minute
    )

    def guard_tool_call(tool: McpTool) -> None:
        if not (tool.annotations or {}).get("readOnlyHint", False):
            mutation_guard.check_rate()

    app = FastAPI(
        title="lm-eval evaluation service",
        version=SERVICE_VERSION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        RequestGateMiddleware,
        guard=mutation_guard,
        max_body_bytes=MAX_MCP_REQUEST_BYTES,
    )

    @app.exception_handler(EvaluationNotFound)
    async def not_found(_request: Request, exc: EvaluationNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)[:4096]})

    @app.exception_handler(EvaluationConflict)
    async def conflict(_request: Request, exc: EvaluationConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)[:4096]})

    @app.exception_handler(ArtifactConflict)
    async def artifact_conflict(
        _request: Request, exc: ArtifactConflict
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)[:4096]})

    @app.exception_handler(ValueError)
    async def invalid(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)[:4096]})

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def ready() -> dict[str, str]:
        try:
            is_ready = service.ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="not ready") from exc
        if not is_ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ready"}

    @app.post("/v1/evaluations:plan", response_model=PlanResponse)
    async def plan(request: EvaluationRequest) -> PlanResponse:
        return service.plan(request)

    @app.post(
        "/v1/evaluations",
        response_model=EvaluationStatus,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit(request: EvaluationRequest) -> EvaluationStatus:
        mutation_guard.check_rate()
        return service.submit(request)

    @app.get("/v1/evaluations", response_model=EvaluationList)
    async def list_evaluations(
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> EvaluationList:
        return service.list(limit)

    @app.get("/v1/evaluations/{evaluation_id}", response_model=EvaluationStatus)
    async def get_evaluation(evaluation_id: str) -> EvaluationStatus:
        return service.get(evaluation_id)

    @app.delete("/v1/evaluations/{evaluation_id}", response_model=EvaluationStatus)
    async def cancel_evaluation(evaluation_id: str) -> EvaluationStatus:
        mutation_guard.check_rate()
        return service.cancel(evaluation_id)

    @app.get("/v1/evaluations/{evaluation_id}/artifacts", response_model=ArtifactList)
    async def list_artifacts(evaluation_id: str) -> ArtifactList:
        return service.list_artifacts(evaluation_id)

    @app.get("/v1/evaluations/{evaluation_id}/report", response_model=EvaluationReport)
    async def get_report(evaluation_id: str) -> EvaluationReport:
        return service.get_report(evaluation_id)

    @app.get(
        "/v1/evaluations/{evaluation_id}/logs",
        response_class=PlainTextResponse,
        responses={200: {"content": {"text/plain": {}}}},
    )
    async def get_logs(
        evaluation_id: str,
        tail_lines: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> str:
        return service.logs(evaluation_id, tail_lines)

    McpServer(
        name="lm-eval-service",
        version=SERVICE_VERSION,
        instructions=(
            "Plan and operate bounded Kubernetes lm-eval profiles. "
            "Use plan_evaluation before submit_evaluation; samples and full logs "
            "remain artifacts."
        ),
        tools=evaluation_tools(service),
        allowed_origins=set(settings.allowed_origins),
        tool_call_guard=guard_tool_call,
    ).install(app)

    return app


def create_configured_app() -> FastAPI:
    token = os.environ.get("LM_EVAL_API_TOKEN", "")
    if not token:
        raise RuntimeError("LM_EVAL_API_TOKEN is required")
    settings = ServiceSettings.from_environment()
    return create_app(settings, KubernetesClient(), token)


def main() -> None:
    uvicorn.run(
        "inference_hard_tools.evaluation_service.api:create_configured_app",
        factory=True,
        host="0.0.0.0",
        port=8080,
        access_log=False,
    )
