"""Kubernetes-native lm-eval service."""

from __future__ import annotations

import os
import re

SERVICE_VERSION = "2.0.0"


def lm_eval_commit() -> str:
    """Return the immutable lm-eval dependency revision baked into the image."""
    value = os.environ.get("LM_EVAL_COMMIT", "")
    if not re.fullmatch(r"[a-f0-9]{40}", value):
        raise RuntimeError(
            "LM_EVAL_COMMIT must be a full dependency commit baked into the image"
        )
    return value
