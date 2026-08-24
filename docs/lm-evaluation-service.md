# lm-eval evaluation service

This service runs operator-approved lm-eval profiles against deployed
OpenAI-compatible models. Each request becomes one CPU-only Kubernetes Job;
the Job and a durable result directory are the operation record. The controller
has no evaluation database or scheduler, and callers cannot provide task files,
dataset URLs, credentials, or arbitrary lm-eval arguments.

## Build and deploy

Both the build base and deployed image must be immutable:

```bash
docker build -f Dockerfile.lm-eval-service \
  --build-arg BASE_IMAGE=python:3.12.11-slim@sha256:<base-digest> \
  --build-arg LM_EVAL_COMMIT=8a07e1110d060de48cfc7a9a7987b7659060b60b \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t registry.example/inference-hard-lm-eval-service:<build-id> .

helm upgrade --install evals charts/lm-eval-service \
  --namespace lm-eval --create-namespace \
  --set image.repository=registry.example/inference-hard-lm-eval-service \
  --set image.digest=sha256:<worker-image-digest> \
  --set auth.existingSecret=lm-eval-api-token
```

The inference-hard-tools image installs `lm-evaluation-harness` from the pinned
upstream commit. It does not patch or vendor lm-eval. Configure `targets` as the
complete operator allowlist; request bodies contain only a logical target name.
Credentials are read from operator-selected Kubernetes Secrets.

## Evaluation profiles

`profiles` is the operator-owned catalog. A request selects a profile by name,
and the complete selected profile participates in the deterministic plan hash.
Each task must name an lm-eval YAML already installed in the pinned image and an
immutable Hugging Face dataset repository revision:

```yaml
profiles:
  gsm8k:
    tasks:
      - name: gsm8k
        config_path: lm_eval/tasks/gsm8k/gsm8k.yaml
        dataset:
          path: openai/gsm8k
          name: main
          revision: 740312add88f781978c0658806c59bc2815b9866
    primary_metric:
      task: gsm8k
      metric: exact_match
      filter: flexible-extract
    default_num_fewshot: 5
    max_num_fewshot: 32
    default_generation_limit: 512
    max_generation_limit: 4096
    max_samples: 1319
```

A profile may contain up to 32 tasks. `primary_metric` must identify one of
them; the report also retains every finite numeric metric produced for every
task. Profile limits bound caller overrides. Updating a task, dataset revision,
credential reference, or bound produces a different plan and evaluation ID.

This supports any leaf task configuration present in the installed lm-eval
version whose dataset can be fixed by repository, configuration name, and
40-character commit hash. Groups, caller-supplied YAML, local paths, unpinned
dataset branches/tags, and arbitrary dataset arguments are deliberately not
accepted.

## Dataset access and caching

Without a shared cache, each Job uses its bounded writable `emptyDir` and may
download its pinned datasets over the worker egress policy. For gated datasets,
set the chart-wide `huggingFace.tokenSecret.name`. Profiles inherit this
reference and may override it with
`dataset_token_secret_name` and `dataset_token_secret_key`. Only evaluation
Jobs receive `HF_TOKEN`; the secret value is never stored in the request, Job
annotations, effective configuration, or report.

For repeatable or disconnected operation, populate a PVC separately with the
exact profile revisions, then configure:

```yaml
datasetCache:
  existingClaim: lm-eval-datasets
  mountPath: /var/cache/lm-eval
```

Workers mount this PVC read-only and set both `HF_HUB_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`. A cache miss therefore fails rather than silently
downloading or mutating shared state. Cache population and lifecycle are
operator workflows outside this service.

Before measurement, the worker resolves each pinned dataset through the same
effective task configuration and records every loaded split's Hugging Face
fingerprint. Reports therefore contain both the declared repository identity
and runtime fingerprints.

## Request and report

All objects reject unknown fields:

```json
{
  "target": "logical-target-name",
  "model": "served-model-name",
  "profile": "gsm8k",
  "num_fewshot": null,
  "limit": null,
  "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
  "generation_limit": null,
  "chat_template": "none",
  "timeout_seconds": 7200
}
```

Null `num_fewshot` and `generation_limit` select profile defaults. The plan
records both requested and resolved values. REST operations remain under
`/v1/evaluations`; evaluation IDs are `eval-` plus 20 lowercase hexadecimal
characters. Plan, submit, list, get, cancel, artifacts, report, and bounded-log
operations have the same lifecycle semantics as the original GSM8K service.

Reports contain generic per-task metric maps, the structured primary metric,
per-task evaluated sample counts, dataset fingerprints, latency, exact
measurement window, lm-eval version and commit, full effective configuration,
served-model and credential-free endpoint identities, terminal state, artifact
hashes, and a bounded error. Samples and raw lm-eval output remain artifacts.

## MCP

`POST /mcp` uses the repository's stateless Streamable HTTP protocol version
`2026-07-28`. Its seven tools are `plan_evaluation`, `submit_evaluation`,
`list_evaluations`, `get_evaluation`, `cancel_evaluation`,
`list_evaluation_artifacts`, and `get_evaluation_report`. Tool schemas are the
same strict models used by REST.

## Limitations

- Durable storage is a shared-filesystem PVC; object-store backends are not
  implemented. Multiple controller replicas require RWX storage.
- The tokenizer-free endpoint mode requires string-prompt support.
- Mutation rate limits are per controller replica.
- Artifact download and cache population are intentionally outside MCP.
- A Kubernetes-successful Job without a durable report exposes a missing report
  rather than guessed metrics.
