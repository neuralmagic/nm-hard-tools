from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nm_hard_tools.model_deployment.renderer import (
    OperatorConfigurationError,
    _installed_manifesto_identity,
)

ROOT = Path(__file__).parents[2]
REVISION = "229195038e506b67bc064bde9f36b6ca13ef5170"


def test_installed_manifesto_identity_uses_direct_url_commit() -> None:
    distribution = Mock(version="0.3.0")
    distribution.read_text.return_value = json.dumps(
        {
            "vcs_info": {
                "vcs": "git",
                "commit_id": REVISION,
                "requested_revision": REVISION,
            }
        }
    )
    with patch(
        "nm_hard_tools.model_deployment.renderer.importlib.metadata.distribution",
        return_value=distribution,
    ):
        assert _installed_manifesto_identity() == f"0.3.0+{REVISION}"


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        {},
        {"vcs_info": {"vcs": "git", "commit_id": "main", "requested_revision": "main"}},
        {
            "vcs_info": {
                "vcs": "git",
                "commit_id": REVISION,
                "requested_revision": "main",
            }
        },
    ],
)
def test_installed_manifesto_identity_rejects_mutable_or_missing_provenance(
    direct_url: dict[str, object] | None,
) -> None:
    distribution = Mock(version="0.3.0")
    distribution.read_text.return_value = (
        None if direct_url is None else json.dumps(direct_url)
    )
    with (
        patch(
            "nm_hard_tools.model_deployment.renderer.importlib.metadata.distribution",
            return_value=distribution,
        ),
        pytest.raises(OperatorConfigurationError, match="immutable VCS provenance"),
    ):
        _installed_manifesto_identity()


def test_model_deployment_image_uses_frozen_lockfile() -> None:
    dockerfile = (ROOT / "Dockerfile.model-deployment").read_text()
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert (
        "uv sync --frozen --no-dev --extra model-deployment --no-editable" in dockerfile
    )
    assert "pip install --no-cache-dir '.[model-deployment]'" not in dockerfile
