#!/usr/bin/env python3
"""Run the structural-feature regression tests before any expensive VF fit."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    raise SystemExit(subprocess.call(
        [sys.executable,"-m","pytest","-q","tests/test_structural_vf.py"],
        cwd=ROOT))
