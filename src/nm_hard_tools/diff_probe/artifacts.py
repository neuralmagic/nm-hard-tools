"""Durable, bounded probe artifact publication and lookup."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from nm_hard_tools.diff_probe.models import (
    ProbeArtifactList,
    validate_probe_id,
)
from nm_hard_tools.evaluation_service.artifacts import (
    ArtifactConflict,
    artifact_metadata,
    atomic_json,
)

__all__ = [
    "MAX_MANIFEST_BYTES",
    "MAX_REPORT_BYTES",
    "ArtifactConflict",
    "ProbeArtifactStore",
    "artifact_metadata",
    "atomic_json",
]

MAX_REPORT_BYTES = 262_144
MAX_MANIFEST_BYTES = 262_144
ARTIFACT_NAMES = (
    "effective-configuration.json",
    "error.txt",
    "samples.jsonl",
    "report.json",
)


class ProbeArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def output_dir(self, probe: str) -> Path:
        validate_probe_id(probe)
        return self.root / probe

    def path(self, probe: str, name: str) -> Path:
        return self.output_dir(probe) / name

    @staticmethod
    def read_json(path: Path, bound: int) -> dict[str, Any]:
        if path.stat().st_size > bound:
            raise ArtifactConflict(
                f"artifact {path.name} exceeds the service response bound"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ArtifactConflict(f"artifact {path.name} is malformed")
        return value

    def list(self, probe: str) -> ProbeArtifactList:
        path = self.path(probe, "artifacts.json")
        report_path = self.path(probe, "report.json")
        if report_path.exists():
            self.ensure_manifest(report_path.parent, probe)
        data: dict[str, Any]
        if path.exists():
            data = self.read_json(path, MAX_MANIFEST_BYTES)
        else:
            data = {"probe_id": probe, "artifacts": []}
        return ProbeArtifactList.model_validate(data)

    def ensure_manifest(self, output_dir: Path, probe: str) -> None:
        """Atomically repair metadata after a terminal report is published."""
        manifest_path = output_dir / "artifacts.json"
        replace_invalid = False
        if manifest_path.exists():
            try:
                ProbeArtifactList.model_validate(
                    self.read_json(manifest_path, MAX_MANIFEST_BYTES)
                )
                return
            except (OSError, ValueError, TypeError):
                replace_invalid = True
        artifact_items = [
            artifact_metadata(candidate)
            for name in ARTIFACT_NAMES
            if (candidate := output_dir / name).exists()
        ]
        artifacts = {"probe_id": probe, "artifacts": artifact_items}
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".artifacts-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(artifacts, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if replace_invalid:
                os.replace(temporary, manifest_path)
            else:
                try:
                    os.link(temporary, manifest_path)
                except FileExistsError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)

    def ready(self) -> bool:
        if not self.root.is_dir():
            return False
        with tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".diff-probe-ready-"
        ) as stream:
            stream.write(b"ready")
            stream.flush()
        return True
