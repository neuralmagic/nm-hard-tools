# NM Hard Tools

Deployable, stateless MCP services for rigorous inference evaluation and
benchmarking. This repository owns the service implementations, container
images, and Helm charts.

Included services:

- a single-tool llm-manifesto model deployment service;
- a profiled `lm-evaluation-harness` controller and worker;
- a bounded MCP adapter for the vLLM forward-pass benchmark service;
- deployment charts for lm-eval, forward-pass, Nyann, and AgentX.

## Development

```bash
uv run --extra test pytest
uv run --extra test ruff format --check src tests
uv run --extra test ruff check src tests
```

Repository governance is stored in `gov/` and validated with
[`govctl`](https://github.com/govctl-org/govctl):

```bash
cargo install govctl --version 0.19.1 --locked
govctl status
govctl check
```

The lm-eval image must be built with an explicit Python base image:

```bash
podman build -f Dockerfile.lm-eval-service \
  --build-arg BASE_IMAGE=python:3.12-slim -t inference-hard-lm-eval:dev .
```

The model deployment service is described in
[`docs/model-deployment-service.md`](docs/model-deployment-service.md).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
