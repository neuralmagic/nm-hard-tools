# Inference Hard Tools

Deployable, stateless MCP services for rigorous inference evaluation and
benchmarking. This repository owns the service implementations and Helm charts;
platforms such as VDP consume its immutable images and chart releases.

Included services:

- a profiled `lm-evaluation-harness` controller and worker;
- a bounded MCP adapter for the vLLM forward-pass benchmark service;
- deployment charts for lm-eval, forward-pass, Nyann, and AgentX.

## Development

```bash
uv run --extra test pytest
uv run --extra test ruff format --check src tests
uv run --extra test ruff check src tests
```

The lm-eval image must be built with an explicit Python base image:

```bash
podman build -f Dockerfile.lm-eval-service \
  --build-arg BASE_IMAGE=python:3.12-slim -t inference-hard-lm-eval:dev .
```
