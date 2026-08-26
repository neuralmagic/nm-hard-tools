from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nm_hard_tools.mcp import PROTOCOL_VERSION, SERVER_INFO_META

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
) -> Any:
    values = dict(params or {})
    values["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
    }
    headers = {
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    if method == "tools/call" and isinstance(values.get("name"), str):
        headers["Mcp-Name"] = values["name"]
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": values},
    )


def test_mcp_identity_and_tool_map(client: TestClient) -> None:
    discovered = rpc(client, "server/discover")
    assert discovered.status_code == 200
    result = discovered.json()["result"]
    assert result["_meta"][SERVER_INFO_META]["name"] == "diff-probe-service"

    listed = rpc(client, "tools/list").json()["result"]["tools"]
    assert {tool["name"] for tool in listed} == {
        "plan_probe",
        "submit_probe",
        "list_probes",
        "get_probe",
        "cancel_probe",
        "get_probe_report",
        "list_probe_artifacts",
    }
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in listed)
    annotations = {tool["name"]: tool["annotations"] for tool in listed}
    assert annotations["plan_probe"]["readOnlyHint"] is True
    assert annotations["cancel_probe"]["destructiveHint"] is True


def test_mcp_plan_and_submit(client: TestClient, request_body: dict[str, Any]) -> None:
    planned = rpc(
        client, "tools/call", {"name": "plan_probe", "arguments": request_body}
    )
    assert planned.status_code == 200
    content = planned.json()["result"]["structuredContent"]
    assert content["probe_id"].startswith("probe-")

    submitted = rpc(
        client, "tools/call", {"name": "submit_probe", "arguments": request_body}
    )
    assert submitted.status_code == 200
    status = submitted.json()["result"]["structuredContent"]
    assert status["probe_id"] == content["probe_id"]
    assert status["state"] == "pending"


def test_mcp_tool_errors_are_bounded(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    response = rpc(
        client,
        "tools/call",
        {"name": "plan_probe", "arguments": {**request_body, "target_b": "nope"}},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert "allowed targets" in result["content"][0]["text"]
