from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inference_hard_tools.mcp import PROTOCOL_VERSION, SERVER_INFO_META

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tests.evaluation_service.conftest import FakeKubernetes


def rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    version: str = PROTOCOL_VERSION,
    headers: dict[str, str] | None = None,
) -> Any:
    values = dict(params or {})
    values["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
    }
    request_headers = {
        "MCP-Protocol-Version": version,
        "Mcp-Method": method,
    }
    if method == "tools/call" and isinstance(values.get("name"), str):
        request_headers["Mcp-Name"] = values["name"]
    request_headers.update(headers or {})
    response = client.post(
        "/mcp",
        headers=request_headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": values},
    )
    assert "mcp-session-id" not in response.headers
    return response


def test_mcp_uses_shared_server_identity(client: TestClient) -> None:
    discovered = rpc(client, "server/discover")
    assert discovered.status_code == 200
    result = discovered.json()["result"]
    assert result["resultType"] == "complete"
    assert result["supportedVersions"] == ["2026-07-28"]
    assert result["_meta"][SERVER_INFO_META]["name"] == "lm-eval-service"


def test_mcp_tools_have_strict_schemas_and_annotations(client: TestClient) -> None:
    response = rpc(client, "tools/list")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["resultType"] == "complete"
    listed = result["tools"]
    assert {tool["name"] for tool in listed} == {
        "plan_evaluation",
        "submit_evaluation",
        "list_evaluations",
        "get_evaluation",
        "cancel_evaluation",
        "list_evaluation_artifacts",
        "get_evaluation_report",
    }
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in listed)
    assert all("outputSchema" in tool for tool in listed)
    annotations = {tool["name"]: tool["annotations"] for tool in listed}
    assert annotations["plan_evaluation"]["readOnlyHint"] is True
    assert annotations["submit_evaluation"]["idempotentHint"] is True
    assert annotations["cancel_evaluation"]["destructiveHint"] is True


def test_mcp_plan_submit_and_stateless_status(
    client: TestClient,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    planned_response = rpc(
        client, "tools/call", {"name": "plan_evaluation", "arguments": request_body}
    )
    assert planned_response.status_code == 200
    planned = planned_response.json()["result"]["structuredContent"]
    assert kube.creates == 0
    submitted = rpc(
        client, "tools/call", {"name": "submit_evaluation", "arguments": request_body}
    ).json()["result"]["structuredContent"]
    assert submitted["evaluation_id"] == planned["evaluation_id"]
    status = rpc(
        client,
        "tools/call",
        {
            "name": "get_evaluation",
            "arguments": {"evaluation_id": planned["evaluation_id"]},
        },
    ).json()["result"]["structuredContent"]
    assert status["state"] == "pending"


def test_mcp_rejects_unknown_arguments(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    result = rpc(
        client,
        "tools/call",
        {
            "name": "plan_evaluation",
            "arguments": {
                **request_body,
                "cli_args": ["--include_path", "https://evil"],
            },
        },
    ).json()["result"]
    assert result["isError"] is True
    assert "Extra inputs are not permitted" in result["content"][0]["text"]


def test_mcp_origin_and_mutation_auth(
    client: TestClient, request_body: dict[str, Any]
) -> None:
    origin = rpc(client, "tools/list", headers={"Origin": "https://attacker.test"})
    assert origin.status_code == 403
    unauthorized = rpc(
        client,
        "tools/call",
        {"name": "submit_evaluation", "arguments": request_body},
        headers={"Authorization": "Bearer wrong"},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == -32001


def test_all_mcp_requests_require_preparse_auth(
    settings: Any, kube: FakeKubernetes
) -> None:
    from fastapi.testclient import TestClient

    from inference_hard_tools.evaluation_service.api import create_app

    untrusted = TestClient(create_app(settings, kube, "operator-token"))
    response = rpc(untrusted, "tools/list")
    assert response.status_code == 401
    assert response.json()["id"] is None


def test_mcp_mutation_rate_limit_remains_an_rpc_error(
    settings: Any,
    kube: FakeKubernetes,
    request_body: dict[str, Any],
) -> None:
    from fastapi.testclient import TestClient

    from inference_hard_tools.evaluation_service.api import create_app

    limited = settings.model_copy(update={"mutation_rate_limit_per_minute": 1})
    client = TestClient(
        create_app(limited, kube, "operator-token"),
        headers={"Authorization": "Bearer operator-token"},
    )
    first = rpc(
        client,
        "tools/call",
        {"name": "submit_evaluation", "arguments": request_body},
    )
    assert first.status_code == 200
    second = rpc(
        client,
        "tools/call",
        {
            "name": "submit_evaluation",
            "arguments": {**request_body, "model": "different-model"},
        },
    )
    assert second.status_code == 429
    assert second.json()["error"]["code"] == -32001


def test_unexpected_tool_exception_stays_in_mcp_envelope(
    client: TestClient, kube: FakeKubernetes
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise OSError("durable storage unavailable")

    kube.list_jobs = fail  # type: ignore[method-assign]
    response = rpc(
        client,
        "tools/call",
        {"name": "list_evaluations", "arguments": {}},
    )
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is True
    assert response.json()["result"]["resultType"] == "complete"
