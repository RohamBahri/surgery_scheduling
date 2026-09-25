from __future__ import annotations

from types import SimpleNamespace

import final_paper_finalization_fixes as hardening
import final_paper_resilience as resilience
import final_paper_runtime_fixes as runtime
import run_final_paper_training as training


def test_resilience_installed_by_supported_wrapper_import_path() -> None:
    resilience.install()
    assert training._training_tie_seed_audit is resilience.safe_training_tie_seed_audit
    assert runtime.safe_saturation_test is resilience.safe_saturation_test
    assert hardening.robust_deterministic_site_solve is resilience.resilient_deterministic_site_solve


def test_tie_seed_diagnostic_failure_is_nonfatal(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("diagnostic only")

    monkeypatch.setattr(resilience, "_ORIGINAL_TIE_AUDIT", boom)
    rows = resilience.safe_training_tie_seed_audit([], None, None)
    assert len(rows) == 1
    assert rows[0]["status"] == "DIAGNOSTIC_FAILED_CONTINUED"
    assert "diagnostic only" in rows[0]["error"]


def test_saturation_diagnostic_failure_is_nonfatal(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("optional saturation")

    monkeypatch.setattr(resilience, "_ORIGINAL_SATURATION", boom)
    rows, status = resilience.safe_saturation_test()
    assert rows == []
    assert status["status"] == "DIAGNOSTIC_FAILED_CONTINUED"
    assert status["policies_already_frozen_before_saturation"] is True


def test_second_emergency_time_limit_keeps_valid_result(monkeypatch) -> None:
    calls = []
    result = SimpleNamespace(diagnostics=SimpleNamespace(status="TIME_LIMIT"))

    def fake_once(*args, **kwargs):
        calls.append(kwargs["wall_seconds"])
        return "view", result

    monkeypatch.setattr(hardening, "_fixed_site_once", fake_once)
    week = SimpleNamespace(position=7)
    settings = SimpleNamespace()
    view, returned = resilience.resilient_deterministic_site_solve(
        week,
        [1.0],
        settings,
        "TGH",
        work_limit=60.0,
        wall_seconds=300,
        mip_gap=0.0,
        seed=42,
    )
    assert view == "view"
    assert returned is result
    assert len(calls) == 2
    assert calls[1] > calls[0]
