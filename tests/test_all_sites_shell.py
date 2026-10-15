"""Integration tests for the all-site shell entrypoint without real data or Gurobi.

These tests verify the exact runner CLI and stage guards; the Python numerical
oracles have their own solver-backed tests in test_anchored_core.py.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "run_all_sites_anchored.sh"
STAGES = ("references", "train", "certify", "evaluate", "sensitivity")


def make_executable(path, content):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + content)
    path.chmod(0o755)


@pytest.fixture
def repo(tmp_path):
    base = tmp_path / "repo with spaces"
    base.mkdir()
    shutil.copy2(SOURCE, base / SOURCE.name)
    (base / "run_weekly_anchored.py").write_text("# Stub runner\n")
    (base / "data").mkdir()
    (base / "data/UHNOperating_RoomScheduling2011-2013.xlsx").write_bytes(b"mock workbook")
    model_dir = base / "artifacts/weekly_rich_v2"
    model_dir.mkdir(parents=True)
    for file in ("learning_settings.json", "model_case.json", "lasso_policy.json",
                 "vf_progress.json"):
        (model_dir / file).write_text("{}")
    gap_dir = base / "artifacts/weekly_gap_v1"
    gap_dir.mkdir(parents=True)
    (gap_dir / "gap_progress.json").write_text("{}")
    mocks = tmp_path / "mocks"
    mocks.mkdir()
    make_executable(mocks / "git", 'if [[ "$1" == branch && "$2" == --show-current ]]; then echo "${MOCK_BRANCH:-anchored-exact-adaptive}"; else exit 3; fi\n')
    make_executable(mocks / "python3.14", 'if [[ "$1" == -c ]]; then exit "${MOCK_IMPORT_EXIT:-0}"; fi\nprintf "%s\\n" "$@" > "$ARGS_CAPTURE"\n')
    make_executable(mocks / "caffeinate", '[[ "$1" == -i ]] || exit 3\nshift\nexec "$@"\n')
    env = dict(os.environ, PATH=str(mocks) + os.pathsep + os.environ["PATH"],
               ARGS_CAPTURE=str(tmp_path / "captured_args.txt"))
    return base, env


def run(base, env, stage, *, dry=False):
    # Executed outside the repository to check script-relative path resolution.
    return subprocess.run(["bash", str(base / SOURCE.name), stage]
                          + (["--dry-run"] if dry else []),
                          cwd=base.parent, env=env, text=True, capture_output=True)


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(SOURCE)], check=True)


@pytest.mark.parametrize("stage", STAGES)
def test_all_stages_execute_with_frozen_flags(repo, stage):
    base, env = repo
    out = base / "artifacts/anchored_ALL_hash_v1"
    out.mkdir(parents=True)
    (out / "anchored_references.json").write_text("{}")
    (out / "anchored_progress.json").write_text('{"complete":true}')
    result = run(base, env, stage)
    assert result.returncode == 0, result.stderr
    args = Path(env["ARGS_CAPTURE"]).read_text().splitlines()
    assert args[:2] == ["run_weekly_anchored.py", "data/UHNOperating_RoomScheduling2011-2013.xlsx"]
    expected = {
        "--artifacts": "artifacts/weekly_rich_v2",
        "--gap-model": "artifacts/weekly_gap_v1/gap_progress.json",
        "--output": "artifacts/anchored_ALL_hash_v1",
        "--site": "all", "--stage": stage,
        "--hash-seed": "0",
        "--max-outer": "15", "--iterations": "20",
        "--closure-rounds": "3", "--exact-sep-batch": "384",
        "--workers": "15", "--threads": "1",
        "--master-threads": "4", "--seconds": "600",
        "--sep-seconds": "90", "--master-seconds": "300",
    }
    assert args[2:] == [value for pair in expected.items() for value in pair]
    assert "--initial-policy" not in args
    assert "--report-gap" not in args


def test_dry_run_outputs_same_args_without_python_or_license(repo):
    base, env = repo
    env["MOCK_IMPORT_EXIT"] = "1"
    result = run(base, env, "references", dry=True)
    assert result.returncode == 0, result.stderr
    assert "--exact-sep-batch 384" in result.stdout
    assert "--site all" in result.stdout
    assert "DRY RUN" in result.stdout
    assert not Path(env["ARGS_CAPTURE"]).exists()


def test_rejects_invalid_stage(repo):
    base, env = repo
    result = run(base, env, "incorrect")
    assert result.returncode == 2 and "Usage:" in result.stderr


def test_rejects_wrong_branch(repo):
    base, env = repo
    env["MOCK_BRANCH"] = "main"
    result = run(base, env, "references")
    assert result.returncode == 2 and "anchored-exact-adaptive" in result.stderr


@pytest.mark.parametrize("stage,prereq", [
    ("train", "anchored_references.json"),
    ("certify", "anchored_progress.json"),
    ("evaluate", "anchored_progress.json"),
    ("sensitivity", "anchored_progress.json"),
])
def test_rejects_out_of_order_stages(repo, stage, prereq):
    base, env = repo
    result = run(base, env, stage)
    assert result.returncode == 2 and ("references" if stage == "train" else "train") in result.stderr


def test_rejects_missing_model_inputs(repo):
    base, env = repo
    (base / "artifacts/weekly_gap_v1/gap_progress.json").unlink()
    result = run(base, env, "references")
    assert result.returncode == 2 and "gap_progress.json" in result.stderr


def test_refuses_missing_python_dependencies(repo):
    base, env = repo
    env["MOCK_IMPORT_EXIT"] = "1"
    result = run(base, env, "references")
    assert result.returncode == 2 and "lacks gurobipy" in result.stderr
    assert not Path(env["ARGS_CAPTURE"]).exists()


def test_spaces_in_absolute_output_path_are_preserved(repo):
    base, env = repo
    dest = base / "output with spaces"
    env["OUT"] = str(dest)
    result = run(base, env, "references")
    assert result.returncode == 0, result.stderr
    args = Path(env["ARGS_CAPTURE"]).read_text().splitlines()
    assert args[args.index("--output") + 1] == str(dest)



def test_rejects_removed_rescue_commands(repo):
    base, env = repo
    for stage in ('rescue-references','finalize-references','diagnose-references'):
        result = run(base, env, stage)
        assert result.returncode == 2
        assert 'Usage:' in result.stderr
