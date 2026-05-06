"""Centralized artifact path management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ArtifactConfig:
    root: Path = Path("artifacts")


class ArtifactRun:
    """Lazily materialized artifact directory for one run."""

    def __init__(self, artifact_root: Path | str, run_kind: str, run_label: str | None = None) -> None:
        self.artifact_root = Path(artifact_root)
        self.run_kind = run_kind
        self.run_label = run_label
        self._run_id = self._build_run_id(run_label)

    @staticmethod
    def _build_run_id(run_label: str | None) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if run_label:
            safe = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in run_label).strip("-")
            if safe:
                return f"{stamp}_{safe}"
        return stamp

    @property
    def run_dir(self) -> Path:
        return self.artifact_root / self.run_kind / self._run_id

    def ensure_run_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def path(self, *parts: str, create_parent: bool = True) -> Path:
        path = self.run_dir.joinpath(*parts)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def directory(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


class ArtifactManager:
    """Factory for scoped artifact runs."""

    def __init__(self, artifact_root: Path | str = Path("artifacts")) -> None:
        self.artifact_root = Path(artifact_root)

    def run(self, run_kind: str, run_label: str | None = None) -> ArtifactRun:
        return ArtifactRun(self.artifact_root, run_kind, run_label)
