"""Bounded REST and stateless Streamable HTTP MCP application for probes."""

from __future__ import annotations

import os
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from nm_hard_tools.diff_probe import SERVICE_VERSION
from nm_hard_tools.diff_probe.artifacts import ArtifactConflict
from nm_hard_tools.diff_probe.config import ProbeServiceSettings
from nm_hard_tools.diff_probe.kube import KubernetesBackend, KubernetesClient
from nm_hard_tools.diff_probe.mcp import probe_tools
from nm_hard_tools.diff_probe.models import (
    ProbeArtifactList,
    ProbeList,
    ProbePlanResponse,
    ProbeReport,
    ProbeRequest,
    ProbeStatus,
)
from nm_hard_tools.diff_probe.service import (
    ProbeConflict,
    ProbeNotFound,
    ProbeService,
)
from nm_hard_tools.evaluation_service.api import (
    MutationGuard,
    RequestGateMiddleware,
)
from nm_hard_tools.mcp import McpServer, McpTool

MAX_MCP_REQUEST_BYTES = 1_048_576


def create_app(
    settings: ProbeServiceSettings,
    kube: KubernetesBackend,
    mutation_token: str,
) -> FastAPI:
    service = ProbeService(settings, kube)
    mutation_guard = MutationGuard(
        mutation_token, settings.mutation_rate_limit_per_minute
    )

    def guard_tool_call(tool: McpTool) -> None:
        if not (tool.annotations or {}).get("readOnlyHint", False):
            mutation_guard.check_rate()

    app = FastAPI(
        title="diff-probe service",
        version=SERVICE_VERSION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        RequestGateMiddleware,
        guard=mutation_guard,
        max_body_bytes=MAX_MCP_REQUEST_BYTES,
        protected_prefix="/v1/probes",
    )

    @app.exception_handler(ProbeNotFound)
    async def not_found(_request: Request, exc: ProbeNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)[:4096]})

    @app.exception_handler(ProbeConflict)
    async def conflict(_request: Request, exc: ProbeConflict) -> JSONResponse:
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

    @app.post("/v1/probes:plan", response_model=ProbePlanResponse)
    async def plan(request: ProbeRequest) -> ProbePlanResponse:
        return service.plan(request)

    @app.post(
        "/v1/probes",
        response_model=ProbeStatus,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit(request: ProbeRequest) -> ProbeStatus:
        mutation_guard.check_rate()
        return service.submit(request)

    @app.get("/v1/probes", response_model=ProbeList)
    async def list_probes(
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ProbeList:
        return service.list(limit)

    @app.get("/v1/probes/{probe_id}", response_model=ProbeStatus)
    async def get_probe(probe_id: str) -> ProbeStatus:
        return service.get(probe_id)

    @app.delete("/v1/probes/{probe_id}", response_model=ProbeStatus)
    async def cancel_probe(probe_id: str) -> ProbeStatus:
        mutation_guard.check_rate()
        return service.cancel(probe_id)

    @app.get("/v1/probes/{probe_id}/artifacts", response_model=ProbeArtifactList)
    async def list_artifacts(probe_id: str) -> ProbeArtifactList:
        return service.list_artifacts(probe_id)

    @app.get("/v1/probes/{probe_id}/report", response_model=ProbeReport)
    async def get_report(probe_id: str) -> ProbeReport:
        return service.get_report(probe_id)

    @app.get(
        "/v1/probes/{probe_id}/logs",
        response_class=PlainTextResponse,
        responses={200: {"content": {"text/plain": {}}}},
    )
    async def get_logs(
        probe_id: str,
        tail_lines: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> str:
        return service.logs(probe_id, tail_lines)

    McpServer(
        name="diff-probe-service",
        version=SERVICE_VERSION,
        instructions=(
            "Plan and operate bounded differential/smoke probes against "
            "allowlisted OpenAI-compatible endpoints. Use plan_probe before "
            "submit_probe; sample bodies and logs remain artifacts."
        ),
        tools=probe_tools(service),
        allowed_origins=set(settings.allowed_origins),
        tool_call_guard=guard_tool_call,
    ).install(app)

    return app


def create_configured_app() -> FastAPI:
    token = os.environ.get("DIFF_PROBE_API_TOKEN", "")
    if not token:
        raise RuntimeError("DIFF_PROBE_API_TOKEN is required")
    settings = ProbeServiceSettings.from_environment()
    return create_app(settings, KubernetesClient(), token)


def main() -> None:
    uvicorn.run(
        "nm_hard_tools.diff_probe.api:create_configured_app",
        factory=True,
        host="0.0.0.0",
        port=8080,
        access_log=False,
    )
