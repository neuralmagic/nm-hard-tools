# Model deployment service

This service exposes one authenticated MCP tool, `deploy_model`, whose only
argument is a complete llm-manifesto model configuration encoded as YAML. It
renders that intent with an operator-pinned Manifesto revision and an
operator-owned cluster profile, creates only new owned Kubernetes resources,
waits for every model pod, and probes the exact inference endpoint returned to
the caller.

The caller cannot select the Kubernetes context, namespace, cluster profile,
credentials, deployment identity, storage roots, routing profile, or submission
policy. YAML aliases, duplicate keys, custom tags, multiple documents,
inheritance, and operator-control fields are rejected before rendering. The
input is limited to 1 MiB and 10,000 parsed nodes.

The initial operator routing policy accepts aggregated topology and renders a
direct vLLM service. Disaggregated and routed topologies are rejected before
cluster mutation until their required RBAC and endpoint policy are defined.

Kueue names the Workload for group `i` of a LeaderWorkerSet
`leaderworkerset-<name>-<i>-<hash>` and stores it in a pod label value, so a
rendered LeaderWorkerSet name longer than 39 characters is admitted by the API
server but never associated with a Workload and never ungated. A LeaderWorkerSet
with 11 or more replicas reaches a two-digit group index and gets one character
less, so the budget is 39 characters minus the width of its widest group index.
Rendering rejects such a name before creating anything, whether or not
the cluster profile sets `kueue.local_queue`, because whether the namespace is
queue managed is a property of the cluster rather than of the rendered object.
A role name the caller can shorten fails as `INVALID_MANIFESTO_CONFIG` and
names that field; a name the caller cannot shorten fails as
`INVALID_OPERATOR_CONFIGURATION` and logs the rendered name, the limit, and the
profile change that fixes it.

## Build and deploy

Build the image for an explicit platform so `TARGETARCH` selects the matching
`kubectl` binary:

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.model-deployment \
  --build-arg BASE_IMAGE=python:3.12-slim@sha256:<base-digest> \
  -t registry.example/nm-hard-model-deployment:<build-id> .
```

Create the bearer-token Secret, then install the chart with an immutable image
and a complete private Manifesto cluster profile:

```bash
kubectl -n hard-tools create secret generic model-deployment-token \
  --from-literal=token='<random-token>'

helm upgrade --install model-deployment charts/model-deployment \
  --namespace hard-tools --create-namespace \
  --set image.repository=registry.example/nm-hard-model-deployment \
  --set image.digest=sha256:<image-digest> \
  --set auth.existingSecret=model-deployment-token \
  --set target.namespace=model-serving \
  --set-file target.clusterProfile=/secure/path/cluster-profile.yaml
```

The chart creates a Role in `target.namespace` and binds it to the controller
ServiceAccount. Set `rbac.create=false` and provide equivalent operator-owned
permissions when the generated resource allowlist is unsuitable.

Workload labels are scoped by Helm release name, so several releases with
different cluster profiles can share a namespace and each Service reaches only
its own controller.

A Deployment's `spec.selector` is immutable, so `helm upgrade` on a release
installed before this scoping fails with `field is immutable`. Delete only the
controller Deployment and upgrade with the same values; Helm recreates it with
the release-scoped selector and leaves the Service and its ClusterIP, the
ConfigMap, the ServiceAccount, the Role binding, and the release history in
place:

```bash
kubectl delete deployment <release>-model-deployment --namespace hard-tools

# Re-run the original install command above, unchanged.
helm upgrade --install <release> charts/model-deployment --namespace hard-tools ...
```

The controller is unreachable between the delete and the replacement pod
passing its readiness probe. `helm uninstall` followed by a fresh install also
works, but it drops the Service ClusterIP and forces every operator-controlled
value, including the cluster profile, to be supplied again.

## MCP call

`POST /mcp` requires `Authorization: Bearer <token>` before the request body is
parsed. The tool input is a closed object:

```json
{
  "manifesto_config": "topology: aggregated\nmodel:\n  id: org/model\n  image_ref: vllm.standard\nruntime: {sidecars: []}\nroles:\n  - name: decode\n    lws: {size: 1}\n    parallelism: {tp: 1, dp: false, ep: false}\n    resources: {cpu: '8', memory: 32Gi, gpus: 1}\n"
}
```

The upstream renderer normally requires `release`. The service accepts it as an
optional compatibility placeholder, removes it from deployment identity input,
and injects the derived identity before rendering. Configurations with the same
parsed model intent and operator render context therefore converge even when
their submitted `release` values or YAML formatting differ. A semantic model
intent change receives a new identity and cannot overwrite the prior deployment.

A successful result contains the exact input digest, stable deployment ID,
target namespace, bounded resource references, and the exact in-cluster
`/v1/models` endpoint that passed the readiness probe. It never returns
the submitted configuration, rendered manifest, logs, or credentials.

Domain failures use a closed bounded recovery object containing `code`,
`message`, `retryable`, `retry_after_ms`, `field_issues`, `current_state`, and
`suggested_action`. The deployment-specific `deployment_id` remains available
after target acceptance so an agent can safely retry the same intent. The tool
advertises `openWorldHint: false` because caller input cannot select a network
destination outside the operator-defined target environment.
