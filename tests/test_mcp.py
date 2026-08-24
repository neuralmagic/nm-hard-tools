from __future__ import annotations

import base64

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_hard_tools.forward_benchmark_mcp import (
    MAX_RESULT_BYTES,
    create_forward_benchmark_mcp_app,
)
from inference_hard_tools.mcp import PROTOCOL_VERSION, McpServer, McpTool, empty_schema


def mcp_request(
    client: TestClient,
    method: str,
    *,
    request_id: int = 1,
    params: dict | None = None,
    version: str = PROTOCOL_VERSION,
    extra_headers: dict[str, str] | None = None,
):
    values = dict(params or {})
    values["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": {"name": "vdp-test", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    headers = {
        "accept": "application/json, text/event-stream",
        "mcp-protocol-version": version,
        "mcp-method": method,
    }
    if method == "tools/call" and isinstance(values.get("name"), str):
        headers["mcp-name"] = values["name"]
    headers.update(extra_headers or {})
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": values},
    )


def test_mcp_2_server_is_stateless_strict_and_has_no_legacy_fallback():
    app = FastAPI()
    McpServer(
        name="test-server",
        version="1.0",
        instructions="Test tools.",
        tools=[
            McpTool(
                name="ping",
                description="Return pong.",
                input_schema=empty_schema(),
                handler=lambda _: {"value": "pong"},
            )
        ],
    ).install(app)
    client = TestClient(app)

    discover = mcp_request(client, "server/discover")
    assert discover.status_code == 200
    assert discover.json()["result"]["supportedVersions"] == [PROTOCOL_VERSION]
    assert discover.json()["result"]["capabilities"] == {"tools": {}}
    assert "mcp-session-id" not in discover.headers

    initialized = mcp_request(client, "initialize")
    assert initialized.status_code == 404
    assert initialized.json()["error"]["code"] == -32601
    session = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": "legacy"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert session.status_code == 400
    missing_metadata = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert missing_metadata.status_code == 400
    assert missing_metadata.json()["error"]["code"] == -32602

    listed = mcp_request(client, "tools/list")
    assert [item["name"] for item in listed.json()["result"]["tools"]] == ["ping"]
    called = mcp_request(client, "tools/call", params={"name": "ping", "arguments": {}})
    assert called.json()["result"]["structuredContent"] == {"value": "pong"}
    encoded_name = "=?base64?" + base64.b64encode(b"ping").decode() + "?="
    encoded = mcp_request(
        client,
        "tools/call",
        params={"name": "ping", "arguments": {}},
        extra_headers={"mcp-name": encoded_name},
    )
    assert encoded.json()["result"]["structuredContent"] == {"value": "pong"}

    unsupported = mcp_request(client, "server/discover", version="2025-11-25")
    assert unsupported.status_code == 400
    assert unsupported.json()["error"] == {
        "code": -32022,
        "message": "Unsupported protocol version",
        "data": {
            "supported": [PROTOCOL_VERSION],
            "requested": "2025-11-25",
        },
    }
    mismatch = mcp_request(
        client,
        "tools/call",
        params={"name": "ping", "arguments": {}},
        extra_headers={"mcp-name": "other"},
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["code"] == -32020
    origin = mcp_request(
        client, "tools/list", extra_headers={"origin": "https://attacker.test"}
    )
    assert origin.status_code == 403
    assert client.get("/mcp").status_code == 405


def test_forward_benchmark_adapter_exposes_bounded_stateless_tools():
    seen: list[tuple[str, str, bytes]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/v1/benchmarks/plan":
            return httpx.Response(200, json={"case_count": 2})
        if request.url.path.endswith("/artifacts/report.json"):
            return httpx.Response(200, json={"schema_version": 10})
        return httpx.Response(404, json={"detail": "not found"})

    client = TestClient(
        create_forward_benchmark_mcp_app(
            "http://benchmark.test", transport=httpx.MockTransport(upstream)
        )
    )
    listed = mcp_request(client, "tools/list")
    tools = {tool["name"]: tool for tool in listed.json()["result"]["tools"]}
    assert set(tools) == {
        "cancel_benchmark",
        "compare_benchmarks",
        "get_benchmark",
        "get_benchmark_report",
        "get_profile_timeline",
        "get_trace_summary",
        "list_benchmark_artifacts",
        "list_benchmarks",
        "plan_benchmark",
        "submit_benchmark",
    }
    assert tools["plan_benchmark"]["annotations"]["readOnlyHint"] is True
    assert tools["cancel_benchmark"]["annotations"]["destructiveHint"] is True

    planned = mcp_request(
        client,
        "tools/call",
        params={
            "name": "plan_benchmark",
            "arguments": {"spec_yaml": "model:\n  id: test/model\n"},
        },
    )
    assert planned.json()["result"]["structuredContent"] == {"case_count": 2}
    assert seen[-1] == (
        "POST",
        "/v1/benchmarks/plan",
        b"model:\n  id: test/model\n",
    )

    report = mcp_request(
        client,
        "tools/call",
        params={
            "name": "get_benchmark_report",
            "arguments": {"job_id": "a" * 24, "name": "report.json"},
        },
    )
    assert report.json()["result"]["structuredContent"] == {"schema_version": 10}
    assert seen[-1][1] == f"/v1/benchmarks/{'a' * 24}/artifacts/report.json"


def test_forward_benchmark_adapter_rejects_unbounded_results():
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * (MAX_RESULT_BYTES + 1),
            headers={"content-type": "text/plain"},
        )

    client = TestClient(
        create_forward_benchmark_mcp_app(
            "http://benchmark.test", transport=httpx.MockTransport(upstream)
        )
    )
    response = mcp_request(
        client,
        "tools/call",
        params={"name": "get_benchmark", "arguments": {"job_id": "b" * 24}},
    )
    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["maximum_bytes"] == MAX_RESULT_BYTES
