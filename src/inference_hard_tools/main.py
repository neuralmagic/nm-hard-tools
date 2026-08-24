from __future__ import annotations

import uvicorn


def forward_benchmark_mcp() -> None:
    uvicorn.run(
        "inference_hard_tools.forward_benchmark_mcp:create_forward_benchmark_mcp_app",
        host="0.0.0.0",
        port=8001,
        factory=True,
    )
