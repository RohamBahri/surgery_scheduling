from __future__ import annotations

import importlib


def test_cli_modules_import() -> None:
    importlib.import_module("scripts.run_experiment")
    importlib.import_module("scripts.run_estimation")
    importlib.import_module("scripts.run_estimation_report")
