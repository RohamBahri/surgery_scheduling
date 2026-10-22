"""Guard the active VF root entrypoints and historical runner relocation."""
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parents[1]

def test_root_runners_are_two_vf_launchers_and_two_shared_libraries():
    expected={"run_selection_12h.sh", "run_vf_followups.sh",
              "run_weekly_rich.py", "run_weekly_train.py"}
    assert {p.name for p in ROOT.glob("run_*") if p.is_file()} == expected

def test_historical_runners_are_importable_from_packages():
    for name in (
        "legacy.weekly.run_weekly",
        "legacy.weekly.run_weekly_diagnostics",
        "legacy.weekly.run_weekly_gap",
        "legacy.anchored.run_weekly_anchored",
        "legacy.anchored.run_anchored_extras",
        "legacy.anchored.run_anchored_research_diagnostics",
    ):
        assert importlib.import_module(name) is not None

def test_original_vf_imports_and_launchers_are_preserved():
    for path in ("pilot/run_theory_full.py","pilot/run_objective_selection.py",
                 "pilot/run_vf_followups.py","run_selection_12h.sh",
                 "run_vf_followups.sh","run_weekly_rich.py","run_weekly_train.py"):
        assert (ROOT/path).is_file()
