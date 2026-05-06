from __future__ import annotations

from pathlib import Path

from src.core.paths import ArtifactManager


def test_artifact_run_layout(tmp_path: Path) -> None:
    run = ArtifactManager(tmp_path).run("experiments", "smoke")
    result_path = run.path("horizon_results.csv")

    assert result_path.parent.exists()
    assert "experiments" in str(result_path)
    assert result_path.name == "horizon_results.csv"


def test_no_import_time_directory_creation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    __import__("src.core.paths")
    assert not (tmp_path / "artifacts").exists()
