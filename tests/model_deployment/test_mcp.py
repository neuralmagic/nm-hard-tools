from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nm_hard_tools.mcp import PROTOCOL_VERSION
from nm_hard_tools.model_deployment.api import create_app
from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.models import ResourceRef
from nm_hard_tools.model_deployment.renderer import (
    RenderedDeployment,
    WorkloadExpectation,
)
from nm_hard_tools.model_deployment.runtime import DeploymentRuntime, RuntimeFailure


class Renderer:
    def render(self, _source: str) -> RenderedDeployment:
        return RenderedDeployment(
            manifesto_digest="a" * 64,
            intent_digest="c" * 64,
            deployment_id="hard-" + "b" * 24,
            namespace="models",
            endpoint="http://model.models.svc.cluster.local:8000/v1/models",
            objects=[
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "model"},
                }
            ],
            resources=[
                ResourceRef(api_version="apps/v1", kind="Deployment", name="model")
            ],
            workloads=[
                WorkloadExpectation(
                    name="model",
                    expected_pods=1,
                    pod_selector={"app.kubernetes.io/instance": "hard-" + "b" * 24},
                )
            ],
        )


class Runtime:
    def deploy(self, _rendered: RenderedDeployment, _timeout: int) -> None:
        return None


class FailingRuntime:
    def deploy(self, _rendered: RenderedDeployment, _timeout: int) -> None:
        raise RuntimeFailure(
            "DEPENDENCY_UNAVAILABLE", "Kubernetes is unavailable", True, True
        )


def client(tmp_path: Path, runtime: DeploymentRuntime | None = None) -> TestClient:
    cluster = tmp_path / "cluster.yaml"
    cluster.write_text("name: test\n")
    token = tmp_path / "token"
    token.write_text("secret")
    settings = DeploymentSettings(
        cluster_profile=cluster,
        namespace="models",
        bearer_token_file=token,
        readiness_timeout_seconds=30,
    )
    return TestClient(
        create_app(
            settings,
            runtime or Runtime(),
            renderer=Renderer(),
            bearer_token="secret",
        )
    )


def request(
    client: TestClient, method: str, params: dict | None = None, token: str = "secret"
):
    values = dict(params or {})
    values["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    headers = {
        "authorization": f"Bearer {token}",
        "mcp-protocol-version": PROTOCOL_VERSION,
        "mcp-method": method,
    }
    if method == "tools/call":
        headers["mcp-name"] = values["name"]
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": values},
    )


def test_authentication_happens_before_manifesto_parsing(tmp_path: Path) -> None:
    response = request(
        client(tmp_path),
        "tools/call",
        {"name": "deploy_model", "arguments": {"manifesto_config": "!unsafe"}},
        token="wrong",
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == -32001
    assert "unsafe" not in response.text


def test_server_advertises_only_the_closed_manifesto_tool(tmp_path: Path) -> None:
    listed = request(client(tmp_path), "tools/list")
    tools = listed.json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["deploy_model"]
    tool = tools[0]
    assert tool["inputSchema"]["additionalProperties"] is False
    assert tool["inputSchema"]["required"] == ["manifesto_config"]
    assert tool["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    assert "oneOf" in tool["outputSchema"]
    assert "ResourceRef" in tool["outputSchema"]["$defs"]
    success_schema = tool["outputSchema"]["$defs"]["DeployModelResult"]
    error_schema = tool["outputSchema"]["$defs"]["DeployModelError"]
    assert set(success_schema["required"]) == set(success_schema["properties"])
    assert set(error_schema["required"]) == set(error_schema["properties"])


def test_discovery_describes_authorization_workflow_and_bounded_results(
    tmp_path: Path,
) -> None:
    discovered = request(client(tmp_path), "server/discover").json()["result"]
    instructions = discovered["instructions"].lower()
    assert "bearer authorization is required and verified" in instructions
    assert "before tool arguments are parsed" in instructions
    assert "deploy_model" in instructions
    assert "operator fixes" in instructions
    assert "bounded" in instructions
    assert "manifests" in instructions


def test_domain_errors_match_the_common_bounded_recovery_shape(tmp_path: Path) -> None:
    response = request(
        client(tmp_path),
        "tools/call",
        {"name": "deploy_model", "arguments": {"manifesto_config": ""}},
    )
    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"] == {
        "schema_version": "1",
        "code": "INVALID_MANIFESTO_CONFIG",
        "message": "manifesto_config must be the only input and fit within 1 MiB",
        "retryable": False,
        "retry_after_ms": None,
        "field_issues": [
            {
                "field": "manifesto_config",
                "code": "INVALID_ARGUMENT",
                "message": "Provide only one Manifesto YAML string up to 1 MiB",
            }
        ],
        "current_state": None,
        "suggested_action": None,
        "deployment_id": None,
    }


def test_retryable_post_acceptance_error_keeps_recovery_identity(
    tmp_path: Path,
) -> None:
    response = request(
        client(tmp_path, FailingRuntime()),
        "tools/call",
        {"name": "deploy_model", "arguments": {"manifesto_config": "release: qwen"}},
    )
    error = response.json()["result"]["structuredContent"]
    assert error["code"] == "DEPENDENCY_UNAVAILABLE"
    assert error["retryable"] is True
    assert error["retry_after_ms"] == 1_000
    assert error["field_issues"] == []
    assert error["current_state"] is None
    assert error["suggested_action"] is None
    assert error["deployment_id"] == "hard-" + "b" * 24


def test_deploy_model_returns_the_probed_endpoint_shape(tmp_path: Path) -> None:
    response = request(
        client(tmp_path),
        "tools/call",
        {"name": "deploy_model", "arguments": {"manifesto_config": "release: qwen"}},
    )
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "schema_version": "1",
        "manifesto_digest": "a" * 64,
        "deployment_id": "hard-" + "b" * 24,
        "endpoint": "http://model.models.svc.cluster.local:8000/v1/models",
        "ready": True,
        "target_namespace": "models",
        "resources": [
            {"api_version": "apps/v1", "kind": "Deployment", "name": "model"}
        ],
    }
