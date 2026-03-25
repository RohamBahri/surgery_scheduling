from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import (
    BlockCalendar,
    BlockId,
    CandidateBlock,
    CaseRecord,
    Col,
    ScheduleAssignment,
    ScheduleResult,
    WeeklyInstance,
)
from src.methods.behavioral_ccg import BehavioralCCGMethod
from src.methods.registry import MethodRegistry


def _train_df() -> pd.DataFrame:
    ts = pd.Timestamp(datetime(2024, 1, 1, 8, 0))
    rows = []
    for i in range(30):
        rows.append(
            {
                Col.CASE_UID: i,
                Col.SURGEON_CODE: "S1",
                Col.CASE_SERVICE: "SvcA",
                Col.PROCEDURE_ID: "P1",
                Col.BOOKED_MINUTES: 100.0,
                Col.PROCEDURE_DURATION: 110.0,
                Col.ACTUAL_START: ts + pd.Timedelta(days=i),
                Col.ACTUAL_STOP: ts + pd.Timedelta(days=i, minutes=110),
                Col.SITE: "TGH",
                Col.OPERATING_ROOM: "OR1",
                Col.MONTH: 1,
                Col.WEEK_OF_YEAR: 1,
                Col.YEAR: 2024,
            }
        )
    return pd.DataFrame(rows)


def _instance() -> WeeklyInstance:
    bid = BlockId(0, "TGH", "OR1")
    case = CaseRecord(1, "P1", "S1", "SvcA", "Elective", "OR1", 100.0, 110.0, datetime(2025, 1, 6, 8, 0), 1, 1, 2025, site="TGH")
    return WeeklyInstance(
        week_index=0,
        start_date=datetime(2025, 1, 6).date(),
        end_date=datetime(2025, 1, 12).date(),
        cases=[case],
        calendar=BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 10.0)]),
        case_eligible_blocks={0: [bid]},
    )


def test_method_fit_plan_cycle(monkeypatch) -> None:
    cfg = Config()
    method = BehavioralCCGMethod(cfg)

    class DummyRec:
        def prepare(self, df):
            return self

        def prepare_instance(self, instance):
            return type("WD", (), {"bookings": np.array([100.0]), "features": np.zeros((1, 1)), "L_bounds": np.array([80.0]), "U_bounds": np.array([120.0]), "surgeon_codes": ["S1"]})()

        def compute_post_review(self, w, wd):
            _ = w, wd
            return np.array([100.0])

    method._recommendation_model = DummyRec()
    method._ccg_result = type("R", (), {"w_optimal": np.zeros(1)})()

    monkeypatch.setattr(
        "src.methods.behavioral_ccg.solve_deterministic",
        lambda **kwargs: ScheduleResult(assignments=[ScheduleAssignment(case_id=1)], solver_status="OPTIMAL"),
    )

    out = method.plan(_instance())
    assert isinstance(out, ScheduleResult)


def test_method_registered_in_runner() -> None:
    registry = MethodRegistry()
    registry.register(BehavioralCCGMethod(Config()))
    assert "BehavioralCCG" in registry.names
