"""Minimal, strict MCP 2.0 Streamable HTTP support for bounded services.

The 2026-07-28 protocol is deliberately stateless: every request carries its
protocol metadata and every HTTP POST is independent. These services do not implement
the legacy initialize/session era and never mints ``Mcp-Session-Id`` values.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

PROTOCOL_VERSION = "2026-07-28"
_PROTOCOL_META = "io.modelcontextprotocol/protocolVersion"
_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO_META = "io.modelcontextprotocol/clientInfo"
SERVER_INFO_META = "io.modelcontextprotocol/serverInfo"
logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


def _decoded_header(value: str | None) -> str | None:
    if value is None or not (value.startswith("=?base64?") and value.endswith("?=")):
        return value
    encoded = value.removeprefix("=?base64?").removesuffix("?=")
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


@dataclass(frozen=True)
class McpToolResult:
    """A tool result with optional structured content and an execution error."""

    text: str
    structured_content: Any | None = None
    is_error: bool = False


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler = field(repr=False, compare=False)
    title: str | None = None
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None

    def definition(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.title:
            value["title"] = self.title
        if self.output_schema:
            value["outputSchema"] = self.output_schema
        if self.annotations:
            value["annotations"] = self.annotations
        return value


class McpServer:
    """A tools-only MCP 2.0 server mounted on an existing FastAPI app."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        instructions: str,
        tools: list[McpTool] | Callable[[], list[McpTool]],
        allowed_origins: set[str] | None = None,
        cache_scope: str = "private",
        tool_call_guard: Callable[[McpTool], None] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.instructions = instructions
        self._tools = tools
        # Non-browser MCP clients omit Origin.  An explicit Origin is denied
        # unless an operator has named it, preventing DNS-rebinding access.
        self.allowed_origins = allowed_origins or set()
        self.cache_scope = cache_scope
        self.tool_call_guard = tool_call_guard

    def tools(self) -> list[McpTool]:
        tools = self._tools() if callable(self._tools) else self._tools
        return sorted(tools, key=lambda item: item.name)

    @property
    def server_meta(self) -> dict[str, Any]:
        return {SERVER_INFO_META: {"name": self.name, "version": self.version}}

    def install(
        self,
        app: FastAPI,
        path: str = "/mcp",
        *,
        dependencies: list[Any] | None = None,
    ) -> None:
        app.add_api_route(
            path,
            self.handle,
            methods=["POST"],
            include_in_schema=False,
            name=f"{self.name}_mcp",
            dependencies=dependencies,
        )

    async def handle(self, request: Request) -> JSONResponse:
        origin = request.headers.get("origin")
        if origin and origin not in self.allowed_origins:
            return self._error(None, -32600, "Origin is not allowed", status=403)
        if request.headers.get("mcp-session-id"):
            return self._error(
                None, -32600, "MCP sessions are not supported", status=400
            )

        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type.lower() != "application/json":
            return self._error(
                None, -32600, "Content-Type must be application/json", status=415
            )
        try:
            message = await request.json()
        except Exception:  # noqa: BLE001 - malformed bytes are a JSON-RPC error.
            return self._error(None, -32700, "Parse error", status=400)
        if not isinstance(message, dict):
            return self._error(None, -32600, "Invalid Request", status=400)

        request_id = message.get("id")
        if (
            message.get("jsonrpc") != "2.0"
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
            or not isinstance(message.get("method"), str)
            or not isinstance(message.get("params", {}), dict)
        ):
            return self._error(None, -32600, "Invalid Request", status=400)

        method = message["method"]
        params = message.get("params", {})
        metadata = params.get("_meta")
        client_info = (
            metadata.get(_CLIENT_INFO_META) if isinstance(metadata, dict) else None
        )
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get(_PROTOCOL_META), str)
            or not isinstance(metadata.get(_CAPABILITIES_META), dict)
            or (
                client_info is not None
                and (
                    not isinstance(client_info, dict)
                    or not isinstance(client_info.get("name"), str)
                    or not isinstance(client_info.get("version"), str)
                )
            )
        ):
            return self._error(
                request_id,
                -32602,
                "Invalid params: required per-request MCP metadata is missing",
                status=400,
            )

        requested_version = metadata[_PROTOCOL_META]
        header_version = request.headers.get("mcp-protocol-version")
        header_method = request.headers.get("mcp-method")
        header_name = _decoded_header(request.headers.get("mcp-name"))
        body_name = params.get("name") if method == "tools/call" else None
        if (
            header_version != requested_version
            or header_method != method
            or (method == "tools/call" and header_name != body_name)
        ):
            return self._error(
                request_id,
                -32020,
                "Header mismatch: MCP request headers do not match the body",
                status=400,
            )
        if requested_version != PROTOCOL_VERSION:
            return self._error(
                request_id,
                -32022,
                "Unsupported protocol version",
                data={
                    "supported": [PROTOCOL_VERSION],
                    "requested": requested_version,
                },
                status=400,
            )

        if method == "server/discover":
            if set(params) != {"_meta"}:
                return self._error(
                    request_id, -32602, "Invalid discover params", status=400
                )
            return self._result(
                request_id,
                {
                    "supportedVersions": [PROTOCOL_VERSION],
                    "capabilities": {"tools": {}},
                    "instructions": self.instructions,
                    "ttlMs": 300_000,
                    "cacheScope": self.cache_scope,
                },
            )
        if method == "tools/list":
            if set(params) - {"_meta", "cursor"}:
                return self._error(
                    request_id,
                    -32602,
                    "Invalid params: this tool list is not paginated",
                    status=400,
                )
            cursor = params.get("cursor")
            if cursor is not None:
                return self._error(
                    request_id,
                    -32602,
                    "Invalid params: this tool list is not paginated",
                    status=400,
                )
            return self._result(
                request_id,
                {
                    "tools": [tool.definition() for tool in self.tools()],
                    "ttlMs": 300_000,
                    "cacheScope": self.cache_scope,
                },
            )
        if method != "tools/call":
            return self._error(
                request_id, -32601, f"Method not found: {method}", status=404
            )

        if set(params) != {"_meta", "name", "arguments"}:
            return self._error(request_id, -32602, "Invalid tool call", status=400)
        arguments = params.get("arguments")
        if not isinstance(body_name, str) or not isinstance(arguments, dict):
            return self._error(request_id, -32602, "Invalid tool call", status=400)
        tool = next((item for item in self.tools() if item.name == body_name), None)
        if tool is None:
            return self._error(
                request_id, -32602, f"Unknown tool: {body_name}", status=400
            )
        if self.tool_call_guard is not None:
            try:
                self.tool_call_guard(tool)
            except HTTPException as error:
                detail = error.detail
                text = detail if isinstance(detail, str) else json.dumps(detail)
                return self._error(
                    request_id,
                    -32001,
                    text,
                    status=error.status_code,
                    headers=error.headers,
                )
        try:
            value = tool.handler(arguments)
            if inspect.isawaitable(value):
                value = await value
        except HTTPException as error:
            detail = error.detail
            text = detail if isinstance(detail, str) else json.dumps(detail)
            return self._tool_result(
                request_id,
                McpToolResult(
                    text=f"HTTP {error.status_code}: {text}",
                    structured_content={"status": error.status_code, "detail": detail},
                    is_error=True,
                ),
            )
        except Exception:
            logger.exception("unexpected MCP tool execution failure: %s", body_name)
            return self._tool_result(
                request_id,
                McpToolResult(text="internal tool execution error", is_error=True),
            )
        if not isinstance(value, McpToolResult):
            value = McpToolResult(
                text=json.dumps(value, default=str, separators=(",", ":")),
                structured_content=value,
            )
        return self._tool_result(request_id, value)

    def _result(self, request_id: str | int, result: dict[str, Any]) -> JSONResponse:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resultType": "complete",
                    **result,
                    "_meta": self.server_meta,
                },
            }
        )

    def _tool_result(
        self, request_id: str | int, result: McpToolResult
    ) -> JSONResponse:
        payload: dict[str, Any] = {
            "content": [{"type": "text", "text": result.text}],
            "isError": result.is_error,
        }
        if result.structured_content is not None:
            payload["structuredContent"] = result.structured_content
        return self._result(request_id, payload)

    @staticmethod
    def _error(
        request_id: str | int | None,
        code: int,
        message: str,
        *,
        status: int,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "error": error},
            status_code=status,
            headers=headers,
        )


def empty_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False}


def fastapi_mcp_tools(
    app: FastAPI,
    *,
    excluded_routes: set[str] | None = None,
    maximum_result_bytes: int = 1_048_576,
) -> list[McpTool]:
    """Expose JSON-compatible ``/v1`` routes as in-process MCP tools.

    FastAPI remains the one implementation of request validation and domain
    behavior.  The MCP adapter translates a tool call into an in-process ASGI
    request, so it neither loops through the network nor duplicates service
    logic. Binary upload/download routes are intentionally excluded by name.
    """

    excluded = excluded_routes or set()
    document = app.openapi()
    definitions = document.get("components", {}).get("schemas", {})

    def rewrite_refs(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    item.replace("#/components/schemas/", "#/$defs/")
                    if key == "$ref" and isinstance(item, str)
                    else rewrite_refs(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite_refs(item) for item in value]
        return value

    rewritten_definitions = rewrite_refs(definitions)

    def referenced_definitions(value: Any) -> dict[str, Any]:
        pending: list[str] = []

        def find_refs(item: Any) -> None:
            if isinstance(item, dict):
                reference = item.get("$ref")
                if isinstance(reference, str) and reference.startswith("#/$defs/"):
                    pending.append(reference.removeprefix("#/$defs/"))
                for nested in item.values():
                    find_refs(nested)
            elif isinstance(item, list):
                for nested in item:
                    find_refs(nested)

        find_refs(value)
        selected: dict[str, Any] = {}
        while pending:
            name = pending.pop()
            if name in selected or name not in rewritten_definitions:
                continue
            selected[name] = rewritten_definitions[name]
            find_refs(selected[name])
        return selected

    tools: list[McpTool] = []
    for route in app.routes:
        if (
            not isinstance(route, APIRoute)
            or not route.path.startswith("/v1/")
            or route.name in excluded
        ):
            continue
        methods = sorted((route.methods or set()) - {"HEAD", "OPTIONS"})
        if len(methods) != 1:
            continue
        method = methods[0]
        operation = document["paths"][route.path][method.lower()]
        properties: dict[str, Any] = {}
        required: list[str] = []
        locations: dict[str, str] = {}
        for parameter in operation.get("parameters", []):
            location = parameter.get("in")
            if location not in {"path", "query"}:
                continue
            name = parameter["name"]
            properties[name] = rewrite_refs(parameter.get("schema", {}))
            locations[name] = location
            if parameter.get("required"):
                required.append(name)

        request_body = operation.get("requestBody")
        body_media_type: str | None = None
        if request_body:
            content = request_body.get("content", {})
            body_media_type = next(
                (
                    media_type
                    for media_type in content
                    if media_type == "application/json" or media_type.endswith("+json")
                ),
                None,
            )
            if body_media_type is None:
                continue
            properties["request"] = rewrite_refs(
                content[body_media_type].get("schema", {"type": "object"})
            )
            if request_body.get("required"):
                required.append("request")

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            input_schema["required"] = required
        tool_definitions = referenced_definitions(input_schema)
        if tool_definitions:
            input_schema["$defs"] = tool_definitions

        async def invoke(
            arguments: dict[str, Any],
            *,
            route_path: str = route.path,
            http_method: str = method,
            parameter_locations: dict[str, str] = locations,
            media_type: str | None = body_media_type,
        ) -> McpToolResult:
            path = route_path
            query: dict[str, Any] = {}
            for name, location in parameter_locations.items():
                if name not in arguments:
                    continue
                if location == "path":
                    path = path.replace(
                        "{" + name + "}", quote(str(arguments[name]), safe="")
                    )
                else:
                    query[name] = arguments[name]
            kwargs: dict[str, Any] = {"params": query}
            if media_type is not None:
                kwargs["json"] = arguments.get("request")
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://inference-hard.internal"
            ) as client:
                response = await client.request(http_method, path, **kwargs)
            if len(response.content) > maximum_result_bytes:
                return McpToolResult(
                    text=(
                        f"Tool result exceeded the {maximum_result_bytes}-byte MCP "
                        "response limit; narrow the request or use the native API."
                    ),
                    is_error=True,
                )
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                structured = response.json()
                text = json.dumps(structured, default=str, separators=(",", ":"))
            else:
                structured = None
                text = response.text
            if response.status_code >= 400:
                text = f"HTTP {response.status_code}: {text}"
            elif response.status_code == 204 and not text:
                text = "Request completed successfully."
                structured = {"status": response.status_code}
            return McpToolResult(
                text=text,
                structured_content=structured,
                is_error=response.status_code >= 400,
            )

        read_only = method == "GET"
        annotations = {
            "readOnlyHint": read_only,
            "destructiveHint": method == "DELETE",
            "idempotentHint": method in {"GET", "DELETE"},
            "openWorldHint": False,
        }
        custom_annotations = operation.get("x-mcp-annotations")
        if isinstance(custom_annotations, dict):
            annotations.update(custom_annotations)
        tools.append(
            McpTool(
                name=route.name,
                title=operation.get("summary") or route.name.replace("_", " ").title(),
                description=(
                    operation.get("description")
                    or f"API operation: {method} {route.path}."
                ),
                input_schema=input_schema,
                handler=invoke,
                annotations=annotations,
            )
        )
    return tools
