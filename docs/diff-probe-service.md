# diff-probe service

Differential correctness and serve-and-smoke probes against allowlisted
OpenAI-compatible endpoints. Each request becomes one CPU-only Kubernetes Job;
the Job and a durable result directory are the operation record. Callers cannot
provide URLs, request bodies, credentials, or sampling parameters — they select
operator-defined logical **targets** and a pinned **request set** by name.

Built as the measurement half of a bug-repro pipeline (for example the RHAIIS
backport process): the service measures and reports; the caller owns thresholds
and verdicts.

## Modes

- **diff** — the same frozen request set, greedy (temperature 0), against
  `target_a` (the configuration under suspicion) and `target_b` (a known-good
  reference). Primary metric `mean_divergence` (1 − similarity ratio, averaged),
  plus `max_divergence`, `exact_match_rate`, and per-side `degenerate_rate_*`
  (empty output, repeated-tail generation, near-single-symbol output). A healthy
  pair reads near zero; the FA4-style garbage-output bug class reads high.
- **smoke** — the request set against a single target with structured checks
  (non-empty, finish reason present, not degenerate). Primary metric
  `pass_rate`.

## Frozen request sets

`request_sets` is the operator-owned catalog: each entry names a JSONL file
below the mounted request-set volume and pins its exact content by sha256. The
worker re-hashes the file before measuring and refuses a moved set — the
request set is part of the measurement's identity. Rows are
`{"messages": [...]}` (chat) or `{"prompt": "..."}` (completions), with an
optional per-row `max_tokens`; vision sets embed images as data URIs so the
set stays self-contained.

## Identity and artifacts

The complete plan (request, request-set pin, endpoint identities, image,
resources) hashes to a deterministic `probe-<hash>` ID; submission is
idempotent and an ID collision with a different plan is a 409. Every run
persists `effective-configuration.json`, `samples.jsonl` (every generated
output), `report.json`, and `artifacts.json` — all content-hashed, so a report
can be audited by a human without trusting the caller's summary of it.

Any transport failure (after bounded retries) fails the probe with a durable
failure report: a partial read is never presented as a measurement.

## Build and deploy

```bash
docker build -f Dockerfile.diff-probe-service \
  --build-arg BASE_IMAGE=python:3.12.11-slim@sha256:<base-digest> \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t registry.example/inference-hard-diff-probe-service:<build-id> .

helm upgrade --install probes charts/diff-probe-service \
  --namespace diff-probe --create-namespace \
  --set image.repository=registry.example/inference-hard-diff-probe-service \
  --set image.digest=sha256:<image-digest> \
  --set auth.existingSecret=diff-probe-api-token \
  --set requestSetStorage.existingClaim=probe-request-sets
```

Configure `targets` (logical name → base URL + optional auth Secret reference)
and `requestSets` (name → file + sha256) as values. The REST surface is
`/v1/probes` (`:plan`, submit, get, cancel, `/report`, `/artifacts`, `/logs`);
mutations require the bearer token. The same operations are exposed as
stateless MCP 2.0 tools on `/mcp` (`plan_probe`, `submit_probe`, `get_probe`,
`get_probe_report`, ...).
