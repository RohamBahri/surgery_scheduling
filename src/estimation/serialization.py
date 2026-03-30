"""Persistence helpers for estimation artifacts."""

from pathlib import Path

import joblib


def save_estimation(result, path: Path) -> Path:
    """Persist estimation artifacts to disk using joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, output_path)
    return output_path


def load_estimation(path: Path):
    """Load persisted estimation artifacts from disk."""
    return joblib.load(Path(path))
