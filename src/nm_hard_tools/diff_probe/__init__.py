"""Differential correctness probe service.

Runs one frozen, operator-owned request set against one or two allowlisted
OpenAI-compatible endpoints as a CPU-only Kubernetes Job, and reports bounded
divergence and degeneracy metrics with content-hashed artifacts.

Two modes:

- ``diff``: the same greedy requests against ``target_a`` (the configuration
  under suspicion) and ``target_b`` (a known-good reference); the primary
  reading is mean output divergence. The probe that demonstrates the
  garbage-output bug class: a healthy pair reads near zero, a broken
  configuration reads high.
- ``smoke``: the request set against a single target with structured
  serve-and-smoke checks (response present, finish reason, no degenerate
  repetition); the primary reading is the pass rate.

The service decides nothing about bugs: it measures. Callers (for example the
backport pipeline's frozen gates) own thresholds and verdicts.
"""

SERVICE_VERSION = "0.1.0"
