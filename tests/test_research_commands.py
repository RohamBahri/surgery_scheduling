"""Confirm the three user-facing research command-line interfaces import cleanly."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "pilot.prepare_data", "pilot.tune", "pilot.experiment", "analysis.report",
])
def test_research_commands_have_help(module):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", module, "--help"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
