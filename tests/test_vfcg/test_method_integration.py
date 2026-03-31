from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.paths import ArtifactManager
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, Col, ScheduleResult, WeeklyInstance
from src.experiments.runner import run_experiment
from src.methods.base import Method
from src.methods.registry import MethodRegistry
from src.methods.vfcg import VFCGMethod
from src.vfcg.types import CertificationResult, VFCGResult


def _minimal_df_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            Col.CASE_UID: [1],
            Col.SURGEON_CODE: ["S1"],
            Col.CASE_SERVICE: ["Svc"],
            Col.PROCEDURE_ID: ["P1"],
            Col.BOOKED_MINUTES: [100.0],
            Col.PROCEDURE_DURATION: [110.0],
            Col.ACTUAL_START: [pd.Timestamp("2024-01-01")],
            Col.ACTUAL_STOP: [pd.Timestamp("2024-01-01 01:50:00")],
            Col.SITE: ["TGH"],
            Col.WEEK_OF_YEAR: [1],
            Col.MONTH: [1],
            Col.YEAR: [2024],
        }
    )


def test_fit_smoke_and_training_week_construction_nonempty(monkeypatch) -> None:
    cfg = Config()
    method = VFCGMethod(cfg)
    captured = {"n_weeks": 0}

    monkeypatch.setattr("src.methods.vfcg.fit_estimation_pipeline", lambda **kwargs: object())

    class _DummyRecModel:
        def __init__(self, **kwargs):
            self.feature_dim = 1

        def prepare(self, df_train):
            return self

        def prepare_instance(self, instance):
            return type("WD", (), {"week_index": instance.week_index, "n_cases": instance.num_cases})()

        def compute_post_review(self, w, week_data):
            return np.array([100.0])

    monkeypatch.setattr("src.methods.vfcg.RecommendationModel", _DummyRecModel)
    monkeypatch.setattr("src.methods.vfcg.apply_experiment_scope", lambda df, config: (df, None))
    monkeypatch.setattr("src.methods.vfcg.build_candidate_pools", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.methods.vfcg.build_eligibility_maps", lambda *args, **kwargs: object())

    class _Inst:
        def __init__(self):
            self.week_index = 0
            self.num_cases = 1

    monkeypatch.setattr("src.methods.vfcg.build_weekly_instance", lambda **kwargs: _Inst())

    def _fake_vfcg_solve(**kwargs):
        captured["n_weeks"] = len(kwargs["week_data_list"])
        return VFCGResult(
            w_optimal=np.array([0.0]),
            objective=1.0,
            n_iterations=1,
            certification=CertificationResult("OPTIMAL_VERIFIED", 0.0, 1.0, 1.0, 1.0, None),
            iterations=[],
            total_cuts_added=0,
        )

    monkeypatch.setattr("src.methods.vfcg.vfcg_solve", _fake_vfcg_solve)

    method.fit(_minimal_df_train())
    assert captured["n_weeks"] >= 1


def test_plan_smoke_uses_post_review_not_realized(monkeypatch) -> None:
    cfg = Config()
    method = VFCGMethod(cfg)

    class _DummyRecModel:
        def prepare_instance(self, instance):
            return type("WD", (), {"bookings": np.array([100.0]), "realized": np.array([999.0])})()

        def compute_post_review(self, w, week_data):
            _ = w, week_data
            return np.array([105.0])

    method._recommendation_model = _DummyRecModel()
    method._vfcg_result = VFCGResult(
        w_optimal=np.array([0.0]),
        objective=0.0,
        n_iterations=0,
        certification=CertificationResult("OPTIMAL_VERIFIED", 0.0, 0.0, 0.0, 0.0, None),
        iterations=[],
        total_cuts_added=0,
    )

    case = CaseRecord(1, "P1", "S1", "Svc", "Elective", "OR1", 100.0, 999.0, datetime(2024, 1, 1), 1, 1, 2024, "TGH")
    cal = BlockCalendar([CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)])
    instance = WeeklyInstance(0, date(2024, 1, 1), date(2024, 1, 7), [case], cal, {0: [cal.candidates[0].id]})

    captured = {"durations": None}

    def _fake_solve_det(**kwargs):
        captured["durations"] = kwargs["durations"]
        return ScheduleResult(assignments=[])

    monkeypatch.setattr("src.methods.vfcg.solve_deterministic", _fake_solve_det)

    _ = method.plan(instance)
    assert np.allclose(captured["durations"], np.array([105.0]))


def test_runner_executes_vfcg_and_writes_summary_artifact(monkeypatch, tmp_path: Path) -> None:
    cfg = Config()
    cfg.data.num_horizons = 1

    warmup = _minimal_df_train()
    pool = _minimal_df_train()

    @dataclass
    class _Scope:
        kept_rows: int = 1

    monkeypatch.setattr("src.experiments.runner.load_data", lambda config: pd.concat([warmup, pool], ignore_index=True))
    monkeypatch.setattr("src.experiments.runner.split_warmup_pool", lambda df, config: (warmup, pool, pd.Timestamp("2024-01-08")))
    monkeypatch.setattr("src.experiments.runner.apply_experiment_scope", lambda df, config: (df, _Scope()))
    monkeypatch.setattr("src.experiments.runner.build_candidate_pools", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.experiments.runner.build_eligibility_maps",
        lambda *args, **kwargs: type("E", (), {"service_rooms": {"Svc": [("TGH", "OR1")]}})(),
    )

    b = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)
    case = CaseRecord(1, "P1", "S1", "Svc", "Elective", "OR1", 100.0, 100.0, datetime(2024, 1, 8), 2, 1, 2024, "TGH")
    wk = WeeklyInstance(0, date(2024, 1, 8), date(2024, 1, 14), [case], BlockCalendar([b]), {0: [b.id]})
    empty = WeeklyInstance(1, date(2024, 1, 15), date(2024, 1, 21), [], BlockCalendar([b]), {})
    calls = {"n": 0}

    def _build_weekly_instance(*args, **kwargs):
        calls["n"] += 1
        return wk if calls["n"] == 1 else empty

    monkeypatch.setattr("src.experiments.runner.build_weekly_instance", _build_weekly_instance)
    monkeypatch.setattr("src.experiments.runner.evaluate", lambda *args, **kwargs: type("K", (), {
        "total_cost": 1.0,
        "activation_cost": 1.0,
        "overtime_cost": 0.0,
        "idle_cost": 0.0,
        "deferral_cost": 0.0,
        "overtime_minutes": 0.0,
        "idle_minutes": 0.0,
        "scheduled_count": 1,
        "deferred_count": 0,
        "blocks_opened": 1,
        "turnover_minutes": 0.0,
    })())
    monkeypatch.setattr("src.experiments.runner.validate_week", lambda *args, **kwargs: None)

    class _DummyVFCG(Method):
        def __init__(self, config):
            super().__init__("VFCG", config)
            self.training_summary = {
                "training_objective": 10.0,
                "training_bound": 9.0,
                "training_gap": 0.1,
                "certification_status": "OPTIMAL_VERIFIED",
                "vfcg_iterations": 2,
                "vfcg_total_cuts": 3,
                "vfcg_max_violation": 0.0,
                "vfcg_tie_break_flags": None,
                "selected_training_weeks": [0],
                "learned_weights": [0.0],
                "master_objective": 10.0,
                "master_bound": 9.0,
            }

        def fit(self, df_train):
            return None

        def plan(self, instance):
            return ScheduleResult(assignments=[], opened_blocks=set(), solver_status="OPTIMAL", objective_value=1.0, diagnostics={})

    reg = MethodRegistry()
    reg.register(_DummyVFCG(cfg))

    artifact_run = ArtifactManager(tmp_path).run("experiments", "vfcg-test")
    out = run_experiment(reg, cfg, artifact_run=artifact_run)
    assert "training_objective" in out.columns
    summary_path = artifact_run.path("vfcg_training_summary.json")
    assert summary_path.exists()
    txt = summary_path.read_text()
    assert "OPTIMAL_VERIFIED" in txt
    assert "selected_training_weeks" in txt


def test_vfcg_method_end_to_end_fit_then_plan_smoke(monkeypatch) -> None:
    cfg = Config()
    method = VFCGMethod(cfg)

    monkeypatch.setattr("src.methods.vfcg.fit_estimation_pipeline", lambda **kwargs: object())

    class _DummyRecModel:
        def __init__(self, **kwargs):
            self.feature_dim = 1

        def prepare(self, df_train):
            return self

        def prepare_instance(self, instance):
            return type("WD", (), {"week_index": instance.week_index, "n_cases": instance.num_cases, "bookings": np.array([100.0])})()

        def compute_post_review(self, w, week_data):
            _ = w, week_data
            return np.array([101.0], dtype=float)

    monkeypatch.setattr("src.methods.vfcg.RecommendationModel", _DummyRecModel)
    monkeypatch.setattr("src.methods.vfcg.apply_experiment_scope", lambda df, config: (df, None))
    monkeypatch.setattr("src.methods.vfcg.build_candidate_pools", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.methods.vfcg.build_eligibility_maps", lambda *args, **kwargs: object())

    class _Inst:
        def __init__(self):
            self.week_index = 0
            self.num_cases = 1

    monkeypatch.setattr("src.methods.vfcg.build_weekly_instance", lambda **kwargs: _Inst())
    monkeypatch.setattr(
        "src.methods.vfcg.vfcg_solve",
        lambda **kwargs: VFCGResult(
            w_optimal=np.array([0.0]),
            objective=1.0,
            n_iterations=1,
            certification=CertificationResult("OPTIMAL_VERIFIED", 0.0, 1.0, 1.0, 1.0, None),
            iterations=[],
            total_cuts_added=0,
        ),
    )
    monkeypatch.setattr("src.methods.vfcg.solve_deterministic", lambda **kwargs: ScheduleResult(assignments=[]))

    method.fit(_minimal_df_train())

    case = CaseRecord(1, "P1", "S1", "Svc", "Elective", "OR1", 100.0, 500.0, datetime(2024, 1, 1), 1, 1, 2024, "TGH")
    cal = BlockCalendar([CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)])
    instance = WeeklyInstance(0, date(2024, 1, 1), date(2024, 1, 7), [case], cal, {0: [cal.candidates[0].id]})
    out = method.plan(instance)
    assert isinstance(out, ScheduleResult)


def test_runner_registration_includes_vfcg_booked_oracle(monkeypatch) -> None:
    from src.cli import run_experiment as cli_mod

    class _DummyMethod:
        def __init__(self, name):
            self.name = name

        def fit(self, df_train):
            return None

        def plan(self, instance):
            return ScheduleResult(assignments=[])

    monkeypatch.setattr(cli_mod, "BookedTimeMethod", lambda config: _DummyMethod("Booked"))
    monkeypatch.setattr(cli_mod, "OracleMethod", lambda config: _DummyMethod("Oracle"))
    monkeypatch.setattr(cli_mod, "VFCGMethod", lambda config: _DummyMethod("VFCG"))

    captured = {"names": None}

    def _fake_run_experiment(registry, config, artifact_run=None):
        captured["names"] = registry.names
        return None

    monkeypatch.setattr(cli_mod, "run_experiment", _fake_run_experiment)
    monkeypatch.setattr("sys.argv", ["prog", "--quick"])
    _ = cli_mod.main()
    assert captured["names"] == ["Booked", "Oracle", "VFCG"]


def test_fallback_path_smoke_sets_training_summary(monkeypatch) -> None:
    cfg = Config()
    method = VFCGMethod(cfg)

    monkeypatch.setattr("src.methods.vfcg.fit_estimation_pipeline", lambda **kwargs: object())

    class _DummyRecModel:
        def __init__(self, **kwargs):
            self.feature_dim = 1

        def prepare(self, df_train):
            return self

        def prepare_instance(self, instance):
            return type("WD", (), {"week_index": instance.week_index, "n_cases": instance.num_cases})()

        def compute_post_review(self, w, week_data):
            return np.array([100.0])

    monkeypatch.setattr("src.methods.vfcg.RecommendationModel", _DummyRecModel)
    monkeypatch.setattr("src.methods.vfcg.apply_experiment_scope", lambda df, config: (df, None))
    monkeypatch.setattr("src.methods.vfcg.build_candidate_pools", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.methods.vfcg.build_eligibility_maps", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.methods.vfcg.build_weekly_instance", lambda **kwargs: type("I", (), {"week_index": 0, "num_cases": 1})())
    monkeypatch.setattr(
        "src.methods.vfcg.vfcg_solve",
        lambda **kwargs: VFCGResult(
            w_optimal=np.array([0.0]),
            objective=float("inf"),
            n_iterations=1,
            certification=CertificationResult("TERMINATED_UNVERIFIED", 1.0, 1.0, 1.0, 1.0, ["fallback"]),
            iterations=[],
            total_cuts_added=0,
        ),
    )

    method.fit(_minimal_df_train())
    summary = method.training_summary
    assert summary["certification_status"] == "TERMINATED_UNVERIFIED"
