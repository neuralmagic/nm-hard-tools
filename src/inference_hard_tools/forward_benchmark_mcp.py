"""Stateless MCP transport adapter for the forward-benchmark service.

The benchmark controller remains the owner of planning, validation, execution,
and artifacts. This sidecar only presents its bounded REST operations as MCP
2.0 tools while the separately released controller image gains native MCP.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints, ValidationError

from inference_hard_tools.mcp import McpServer, McpTool, McpToolResult

MAX_RESULT_BYTES = 1_048_576
JobId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{24}$")]
ArtifactName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
]


class BenchmarkSpecInput(BaseModel):
    spec_yaml: str = Field(
        min_length=1,
        max_length=1_048_576,
        description="Complete benchmark specification encoded as YAML.",
    )


class BenchmarkRef(BaseModel):
    job_id: JobId


class BenchmarkFilters(BaseModel):
    state: str | None = None
    accelerator: str | None = None
    query: str | None = None
    label: str | None = None
    fingerprint: str | None = None


class ReportName(StrEnum):
    REPORT_JSON = "report.json"
    REPORT_MARKDOWN = "report.md"
    SPEC = "spec.yaml"
    MANIFEST = "artifact-manifest.json"


class BenchmarkReportRef(BenchmarkRef):
    name: ReportName = ReportName.REPORT_JSON


class TraceRef(BaseModel):
    job_id: JobId
    trace: ArtifactName
    family: str | None = None


class BenchmarkComparison(BaseModel):
    job_ids: list[JobId] = Field(min_length=2, max_length=8)


def _validated(model: type[BaseModel], arguments: dict[str, Any]) -> BaseModel:
    try:
        return model.model_validate(arguments)
    except ValidationError as error:
        raise HTTPException(422, detail=error.errors(include_url=False)) from error


def create_forward_benchmark_mcp_app(
    upstream_url: str | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the sidecar app; injectable transport keeps its contract testable."""

    upstream = (
        upstream_url or os.environ.get("INFERENCE_HARD_FORWARD_MCP_UPSTREAM")
    ) or "http://127.0.0.1:8000"
    upstream = upstream.rstrip("/")
    app = FastAPI(title="vLLM Forward Benchmark MCP Adapter", version="0.1.0")

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    async def request(
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> McpToolResult:
        headers = {"content-type": "application/yaml"} if content is not None else {}
        try:
            async with httpx.AsyncClient(
                base_url=upstream,
                transport=transport,
                timeout=30,
                follow_redirects=False,
            ) as client:
                response = await client.request(
                    method, path, params=params, content=content, headers=headers
                )
        except httpx.HTTPError as error:
            return McpToolResult(
                text=f"Benchmark service request failed: {error}", is_error=True
            )
        body = response.content
        if len(body) > MAX_RESULT_BYTES:
            return McpToolResult(
                text="Benchmark service result exceeds the 1 MiB MCP result limit",
                structured_content={
                    "status": response.status_code,
                    "size_bytes": len(body),
                    "maximum_bytes": MAX_RESULT_BYTES,
                },
                is_error=True,
            )
        content_type = response.headers.get("content-type", "")
        value: Any
        if "json" in content_type:
            try:
                value = response.json()
            except ValueError:
                value = {"detail": body.decode("utf-8", errors="replace")}
        else:
            value = body.decode("utf-8", errors="replace")
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, default=str, separators=(",", ":"))
        )
        if response.is_error:
            return McpToolResult(
                text=f"Benchmark service returned HTTP {response.status_code}: {text}",
                structured_content={"status": response.status_code, "detail": value},
                is_error=True,
            )
        return McpToolResult(
            text=text,
            structured_content=value if isinstance(value, (dict, list)) else None,
        )

    async def plan(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkSpecInput, arguments)
        return await request("POST", "/v1/benchmarks/plan", content=values.spec_yaml)

    async def submit(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkSpecInput, arguments)
        return await request("POST", "/v1/benchmarks", content=values.spec_yaml)

    async def list_benchmarks(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkFilters, arguments)
        params = values.model_dump(exclude_none=True)
        if "query" in params:
            params["q"] = params.pop("query")
        return await request("GET", "/v1/benchmarks", params=params)

    async def get_benchmark(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkRef, arguments)
        return await request("GET", f"/v1/benchmarks/{values.job_id}")

    async def cancel_benchmark(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkRef, arguments)
        return await request("DELETE", f"/v1/benchmarks/{values.job_id}")

    async def list_artifacts(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkRef, arguments)
        return await request("GET", f"/v1/benchmarks/{values.job_id}/artifacts")

    async def get_report(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkReportRef, arguments)
        name = quote(values.name.value, safe="")
        return await request("GET", f"/v1/benchmarks/{values.job_id}/artifacts/{name}")

    async def trace_summary(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(TraceRef, arguments)
        params = {"run": values.job_id, "trace": values.trace}
        if values.family is not None:
            params["family"] = values.family
        return await request("GET", "/v1/trace-summary", params=params)

    async def profile_timeline(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(TraceRef, arguments)
        params = {"run": values.job_id, "trace": values.trace}
        if values.family is not None:
            params["family"] = values.family
        return await request("GET", "/v1/profile-timeline", params=params)

    async def compare(arguments: dict[str, Any]) -> McpToolResult:
        values = _validated(BenchmarkComparison, arguments)
        return await request(
            "GET", "/v1/compare", params={"runs": ",".join(values.job_ids)}
        )

    read_only = {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    mutating = {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    destructive = {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    tools = [
        McpTool(
            "plan_benchmark",
            "Validate and preview a benchmark specification without submitting it.",
            BenchmarkSpecInput.model_json_schema(),
            plan,
            annotations=read_only,
        ),
        McpTool(
            "submit_benchmark",
            "Validate and submit a benchmark specification to the cluster queue.",
            BenchmarkSpecInput.model_json_schema(),
            submit,
            annotations=mutating,
        ),
        McpTool(
            "list_benchmarks",
            "List benchmark runs with optional filters.",
            BenchmarkFilters.model_json_schema(),
            list_benchmarks,
            annotations=read_only,
        ),
        McpTool(
            "get_benchmark",
            "Get the specification, state, and result metadata for one run.",
            BenchmarkRef.model_json_schema(),
            get_benchmark,
            annotations=read_only,
        ),
        McpTool(
            "cancel_benchmark",
            "Cancel one non-terminal benchmark run.",
            BenchmarkRef.model_json_schema(),
            cancel_benchmark,
            annotations=destructive,
        ),
        McpTool(
            "list_benchmark_artifacts",
            "List the immutable artifacts produced by one benchmark run.",
            BenchmarkRef.model_json_schema(),
            list_artifacts,
            annotations=read_only,
        ),
        McpTool(
            "get_benchmark_report",
            "Read a bounded report, specification, or artifact manifest.",
            BenchmarkReportRef.model_json_schema(),
            get_report,
            annotations=read_only,
        ),
        McpTool(
            "get_trace_summary",
            "Read a bounded kernel summary for one benchmark profile trace.",
            TraceRef.model_json_schema(),
            trace_summary,
            annotations=read_only,
        ),
        McpTool(
            "get_profile_timeline",
            "Read the bounded representative GPU timeline for one trace.",
            TraceRef.model_json_schema(),
            profile_timeline,
            annotations=read_only,
        ),
        McpTool(
            "compare_benchmarks",
            "Compare structured reports for compatible benchmark runs.",
            BenchmarkComparison.model_json_schema(),
            compare,
            annotations=read_only,
        ),
    ]
    McpServer(
        name="vllm-forward-benchmark",
        version="0.1.0",
        instructions=(
            "Plan before submitting. Inspect requested GPU resources and queue "
            "placement, then use bounded reports and summaries instead of raw traces."
        ),
        tools=tools,
    ).install(app)
    return app
