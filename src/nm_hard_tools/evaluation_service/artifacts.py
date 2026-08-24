"""Durable, bounded evaluation artifact publication and lookup."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from nm_hard_tools.evaluation_service.models import (
    ArtifactList,
    validate_evaluation_id,
)

MAX_REPORT_BYTES = 262_144
MAX_MANIFEST_BYTES = 262_144
ARTIFACT_NAMES = (
    "effective-configuration.json",
    "error.txt",
    "lm-eval-result.json",
    "samples.jsonl",
    "worker.log",
    "report.json",
)
MEDIA_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".log": "text/plain",
    ".txt": "text/plain",
}


class ArtifactConflict(RuntimeError):
    """A durable artifact cannot safely satisfy the requested operation."""


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact atomically after flushing its temporary file."""
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}-",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_metadata(path: Path, media_type: str | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": path.name,
        "media_type": media_type
        or MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def output_dir(self, evaluation: str) -> Path:
        validate_evaluation_id(evaluation)
        return self.root / evaluation

    def path(self, evaluation: str, name: str) -> Path:
        return self.output_dir(evaluation) / name

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

    def list(self, evaluation: str) -> ArtifactList:
        path = self.path(evaluation, "artifacts.json")
        report_path = self.path(evaluation, "report.json")
        if report_path.exists():
            self.ensure_manifest(report_path.parent, evaluation)
        data: dict[str, Any]
        if path.exists():
            data = self.read_json(path, MAX_MANIFEST_BYTES)
        else:
            data = {"evaluation_id": evaluation, "artifacts": []}
        log_path = self.path(evaluation, "worker.log")
        known = {item.get("name") for item in data.get("artifacts", [])}
        if log_path.exists() and log_path.name not in known:
            data.setdefault("artifacts", []).append(artifact_metadata(log_path))
        return ArtifactList.model_validate(data)

    def ensure_manifest(self, output_dir: Path, evaluation: str) -> None:
        """Atomically repair metadata after a terminal report is published."""
        manifest_path = output_dir / "artifacts.json"
        replace_invalid = False
        if manifest_path.exists():
            try:
                ArtifactList.model_validate(
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
        artifacts = {"evaluation_id": evaluation, "artifacts": artifact_items}
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

    def tail_log(self, evaluation: str, tail_lines: int) -> str:
        path = self.path(evaluation, "worker.log")
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65_536))
            durable = stream.read().decode("utf-8", errors="replace")
        return "\n".join(durable.splitlines()[-tail_lines:])[-65_536:]

    def ready(self) -> bool:
        if not self.root.is_dir():
            return False
        with tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".lm-eval-ready-"
        ) as stream:
            stream.write(b"ready")
            stream.flush()
        return True
