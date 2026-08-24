# VDP integration

VDP consumes the versioned chart archives from this repository and selects
digest-pinned service images through its image BOM. The chart release is pinned
in VDP independently from VDP's own controller release.

For local development, pass explicit chart paths:

```bash
vdp install --namespace dev-alice \
  --forward-pass-benchmark-chart ../inference-hard-tools/charts/forward-pass-benchmark \
  --lm-eval-chart ../inference-hard-tools/charts/lm-eval-service \
  --nyann-chart ../inference-hard-tools/charts/nyann-bench \
  --agentx-chart ../inference-hard-tools/charts/agentx-service
```

The VDP operator-bootstrap chart deliberately retains the namespaced RBAC it
delegates to these releases. Each chart supports `rbac.create=false` so the
namespace owner does not recreate operator-owned grants.
