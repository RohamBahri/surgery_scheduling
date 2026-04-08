"""Recommendation-model bridge from estimation artifacts to planning durations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.config import CostConfig
from src.core.types import BlockCalendar, BlockId, CaseRecord, Col, WeeklyInstance
from src.estimation import EstimationResult


@dataclass
class SOS2CaseData:
    case_index: int
    profile_id: int
    knot_x: np.ndarray
    knot_y: np.ndarray
    booking: float
    L_bound: float
    U_bound: float


@dataclass
class DiscreteDisplayCaseData:
    case_index: int
    profile_id: int
    booking: float
    L_bound: float
    U_bound: float
    grid_values: np.ndarray
    delta_rec_lb: np.ndarray
    delta_rec_ub: np.ndarray


@dataclass
class WeekRecommendationData:
    week_index: int
    n_cases: int
    features: np.ndarray
    bookings: np.ndarray
    realized: np.ndarray
    L_bounds: np.ndarray
    U_bounds: np.ndarray
    surgeon_codes: List[str]
    sos2_data: List[SOS2CaseData]
    case_eligible_blocks: Dict[int, List[BlockId]]
    calendar: BlockCalendar
    discrete_display_data: List[DiscreteDisplayCaseData] = field(default_factory=list)


class RecommendationModel:
    """Computes recommended and post-review durations given weights w."""

    def __init__(
        self,
        estimation_result: EstimationResult,
        costs: CostConfig,
        plausibility_tails: Tuple[float, float] = (0.01, 0.99),
        w_max: float = 10.0,
    ):
        self._estimation = estimation_result
        self._costs = costs
        self._tau_L, self._tau_U = plausibility_tails
        self._w_max = w_max
        self._feature_names: List[str] = []
        self._procedure_levels: List[str] = []
        self._service_levels: List[str] = []
        self._surgeon_levels: List[str] = []
        self._week_levels: List[int] = []
        self._month_levels: List[int] = []
        self._year_levels: List[int] = []
        self._is_prepared = False
        self._display_grid_step: float = 5.0
        self._display_grid_residue: float = 0.0

    @property
    def feature_dim(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def prepare(self, df_train: pd.DataFrame) -> "RecommendationModel":
        work = df_train.copy()

        if Col.PROCEDURE_ID in work.columns:
            self._procedure_levels = self._sorted_str_levels(work[Col.PROCEDURE_ID])
        else:
            self._procedure_levels = []

        if Col.CASE_SERVICE in work.columns:
            self._service_levels = self._sorted_str_levels(work[Col.CASE_SERVICE])
        else:
            self._service_levels = []

        if Col.SURGEON_CODE in work.columns:
            self._surgeon_levels = self._sorted_str_levels(work[Col.SURGEON_CODE])
        else:
            self._surgeon_levels = []

        if Col.WEEK_OF_YEAR in work.columns:
            self._week_levels = self._sorted_int_levels(work[Col.WEEK_OF_YEAR])
        else:
            self._week_levels = []

        if Col.MONTH in work.columns:
            self._month_levels = self._sorted_int_levels(work[Col.MONTH])
        else:
            self._month_levels = []

        if Col.YEAR in work.columns:
            self._year_levels = self._sorted_int_levels(work[Col.YEAR])
        else:
            self._year_levels = []

        if Col.BOOKED_MINUTES in work.columns:
            bookings = pd.to_numeric(work[Col.BOOKED_MINUTES], errors="coerce").dropna().to_numpy(dtype=float)
            if bookings.size > 0:
                residues = np.mod(bookings, self._display_grid_step)
                self._display_grid_residue = self._normalize_residue(float(np.median(residues)))

        train_cases = self._df_to_cases(work)
        _ = self.build_features(train_cases)
        self._is_prepared = True
        return self

    def prepare_instance(self, instance: WeeklyInstance) -> WeekRecommendationData:
        if not self._is_prepared:
            self._procedure_levels = self._sorted_str_values(c.procedure_id for c in instance.cases)
            self._service_levels = self._sorted_str_values(c.service for c in instance.cases)
            self._surgeon_levels = self._sorted_str_values(c.surgeon_code for c in instance.cases)
            self._week_levels = self._sorted_int_values(c.week_of_year for c in instance.cases)
            self._month_levels = self._sorted_int_values(c.month for c in instance.cases)
            self._year_levels = self._sorted_int_values(c.year for c in instance.cases)

            bookings = np.array([c.booked_duration_min for c in instance.cases], dtype=float)
            if bookings.size > 0:
                residues = np.mod(bookings, self._display_grid_step)
                self._display_grid_residue = self._normalize_residue(float(np.median(residues)))
            _ = self.build_features(instance.cases)
            self._is_prepared = True

        features = self.build_features(instance.cases)
        bookings = np.array([c.booked_duration_min for c in instance.cases], dtype=float)
        realized = np.array([c.actual_duration_min for c in instance.cases], dtype=float)
        L_bounds, U_bounds = self.compute_plausibility_bounds(instance.cases)
        surgeon_codes = [c.surgeon_code for c in instance.cases]

        week_data = WeekRecommendationData(
            week_index=instance.week_index,
            n_cases=instance.num_cases,
            features=features,
            bookings=bookings,
            realized=realized,
            L_bounds=L_bounds,
            U_bounds=U_bounds,
            surgeon_codes=surgeon_codes,
            sos2_data=[],
            case_eligible_blocks=instance.case_eligible_blocks,
            calendar=instance.calendar,
            discrete_display_data=[],
        )
        week_data.sos2_data = self.build_sos2_data(week_data)
        week_data.discrete_display_data = self.build_discrete_display_data(week_data)
        return week_data

    def build_features(self, cases: Sequence[CaseRecord]) -> np.ndarray:
        target_ratio = self._costs.overtime_per_minute / (
            self._costs.overtime_per_minute + self._costs.idle_per_minute
        )

        if not self._feature_names:
            self._feature_names = (
                ["bias", "misalignment_s"]
                + [f"procedure::{p}" for p in self._procedure_levels[1:]]
                + [f"service::{s}" for s in self._service_levels[1:]]
                + [f"surgeon::{s}" for s in self._surgeon_levels[1:]]
                + [f"week::{w}" for w in self._week_levels[1:]]
                + [f"month::{m}" for m in self._month_levels[1:]]
                + [f"year::{y}" for y in self._year_levels[1:]]
            )

        rows: List[List[float]] = []
        for case in cases:
            q_hat = float(self._estimation.critical_ratios.get_ratio(case.surgeon_code))
            misalignment = q_hat - target_ratio
            row = [1.0, misalignment]
            row += self._drop_first_one_hot(str(case.procedure_id), self._procedure_levels)
            row += self._drop_first_one_hot(str(case.service), self._service_levels)
            row += self._drop_first_one_hot(str(case.surgeon_code), self._surgeon_levels)
            row += self._drop_first_one_hot(int(case.week_of_year), self._week_levels)
            row += self._drop_first_one_hot(int(case.month), self._month_levels)
            row += self._drop_first_one_hot(int(case.year), self._year_levels)
            rows.append(row)

        if not rows:
            return np.zeros((0, len(self._feature_names)), dtype=float)
        return np.array(rows, dtype=float)

    def _sorted_str_levels(self, values: pd.Series) -> List[str]:
        series = values.dropna().astype(str).str.strip()
        series = series[series != ""]
        return sorted(series.unique().tolist())

    def _sorted_int_levels(self, values: pd.Series) -> List[int]:
        series = pd.to_numeric(values, errors="coerce").dropna().astype(int)
        return sorted(series.unique().tolist())

    def _sorted_str_values(self, values: Sequence[object]) -> List[str]:
        cleaned = []
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text == "":
                continue
            cleaned.append(text)
        return sorted(set(cleaned))

    def _sorted_int_values(self, values: Sequence[object]) -> List[int]:
        cleaned = []
        for value in values:
            if pd.isna(value):
                continue
            cleaned.append(int(value))
        return sorted(set(cleaned))

    def _drop_first_one_hot(self, value: object, levels: Sequence[object]) -> List[float]:
        if len(levels) <= 1:
            return []
        return [1.0 if value == level else 0.0 for level in levels[1:]]

    def compute_plausibility_bounds(self, cases: Sequence[CaseRecord]) -> Tuple[np.ndarray, np.ndarray]:
        case_df = self._cases_to_quantile_df(cases)
        L = np.asarray(self._estimation.quantile_model.predict(case_df, q=self._tau_L), dtype=float)
        U = np.asarray(self._estimation.quantile_model.predict(case_df, q=self._tau_U), dtype=float)

        bookings = np.array([c.booked_duration_min for c in cases], dtype=float)
        L = np.minimum(L, bookings)
        U = np.maximum(U, bookings)
        return L, U

    def compute_corrections(
        self,
        w: np.ndarray,
        X: np.ndarray,
        bookings: np.ndarray,
        L: np.ndarray,
        U: np.ndarray,
    ) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        if len(w) != X.shape[1]:
            raise ValueError(f"weight dimension mismatch: got {len(w)}, expected {X.shape[1]}")
        if np.any(np.abs(w) > self._w_max + 1e-9):
            raise ValueError(f"weights exceed w_max={self._w_max}")

        delta_rec = X @ w
        return np.clip(delta_rec, L - bookings, U - bookings)

    def apply_response(self, delta_rec: np.ndarray, surgeon_codes: Sequence[str]) -> np.ndarray:
        delta_post = np.zeros_like(delta_rec, dtype=float)
        for i, surgeon in enumerate(surgeon_codes):
            profile = self._get_profile(surgeon)
            u = float(delta_rec[i])
            if u < -profile.h_minus:
                delta_post[i] = profile.a * (u + profile.h_minus)
            elif u > profile.h_plus:
                delta_post[i] = profile.a * (u - profile.h_plus)
            else:
                delta_post[i] = 0.0
        return delta_post

    def compute_post_review(self, w: np.ndarray, week_data: WeekRecommendationData) -> np.ndarray:
        delta_rec = self.compute_corrections(
            w=w,
            X=week_data.features,
            bookings=week_data.bookings,
            L=week_data.L_bounds,
            U=week_data.U_bounds,
        )
        delta_post = self.apply_response(delta_rec=delta_rec, surgeon_codes=week_data.surgeon_codes)
        d_cont = week_data.bookings + delta_post

        if len(week_data.discrete_display_data) != week_data.n_cases:
            week_data.discrete_display_data = self.build_discrete_display_data(week_data)

        d_disc = np.empty_like(d_cont, dtype=float)
        for i, target in enumerate(d_cont):
            case_data = week_data.discrete_display_data[i]
            idx = self._nearest_grid_index(
                grid_values=case_data.grid_values,
                target=float(target),
                booking=float(week_data.bookings[i]),
            )
            d_disc[i] = float(case_data.grid_values[idx])
        return d_disc

    def build_sos2_data(self, week_data: WeekRecommendationData) -> List[SOS2CaseData]:
        if self._estimation.response_profiler is None:
            raise RuntimeError("response_profiler is required to build SOS2 knots")

        out: List[SOS2CaseData] = []
        for i in range(week_data.n_cases):
            surgeon = week_data.surgeon_codes[i]
            pid = int(self._estimation.response_profiler.get_profile_id(surgeon))
            x, y = self._estimation.response_profiler.get_sos2_knots(
                profile_id=pid,
                L_ti=float(week_data.L_bounds[i]),
                U_ti=float(week_data.U_bounds[i]),
                b_ti=float(week_data.bookings[i]),
            )
            out.append(
                SOS2CaseData(
                    case_index=i,
                    profile_id=pid,
                    knot_x=np.asarray(x, dtype=float),
                    knot_y=np.asarray(y, dtype=float),
                    booking=float(week_data.bookings[i]),
                    L_bound=float(week_data.L_bounds[i]),
                    U_bound=float(week_data.U_bounds[i]),
                )
            )
        return out

    def build_discrete_display_data(self, week_data: WeekRecommendationData) -> List[DiscreteDisplayCaseData]:
        out: List[DiscreteDisplayCaseData] = []
        for i in range(week_data.n_cases):
            surgeon = week_data.surgeon_codes[i]
            profile = self._get_profile(surgeon)
            pid = int(getattr(profile, "profile_id", 0))
            booking = float(week_data.bookings[i])
            L_bound = float(week_data.L_bounds[i])
            U_bound = float(week_data.U_bounds[i])
            global_lb = L_bound - booking
            global_ub = U_bound - booking

            grid_values = self._make_display_grid(
                booking=booking,
                L_bound=L_bound,
                U_bound=U_bound,
            )

            feasible_grid: List[float] = []
            pre_lb: List[float] = []
            pre_ub: List[float] = []
            for g_idx, g_val in enumerate(grid_values):
                d_low, d_high = self._display_cell_bounds(
                    grid_values=grid_values,
                    grid_index=g_idx,
                    L_bound=L_bound,
                    U_bound=U_bound,
                )
                interval = self._delta_rec_interval_for_display_cell(
                    booking=booking,
                    d_low=d_low,
                    d_high=d_high,
                    a=float(getattr(profile, "a", 0.0)),
                    h_plus=float(getattr(profile, "h_plus", 0.0)),
                    h_minus=float(getattr(profile, "h_minus", 0.0)),
                    global_lb=global_lb,
                    global_ub=global_ub,
                )
                if interval is None:
                    continue
                lb, ub = interval
                if ub + 1e-9 < lb:
                    continue
                feasible_grid.append(float(g_val))
                pre_lb.append(float(lb))
                pre_ub.append(float(ub))

            if not feasible_grid:
                feasible_grid = [booking]
                pre_lb = [global_lb]
                pre_ub = [global_ub]

            out.append(
                DiscreteDisplayCaseData(
                    case_index=i,
                    profile_id=pid,
                    booking=booking,
                    L_bound=L_bound,
                    U_bound=U_bound,
                    grid_values=np.asarray(feasible_grid, dtype=float),
                    delta_rec_lb=np.asarray(pre_lb, dtype=float),
                    delta_rec_ub=np.asarray(pre_ub, dtype=float),
                )
            )
        return out

    def predict_at_quantile(self, week_data: WeekRecommendationData, q: float) -> np.ndarray:
        case_df = self._cases_to_quantile_df(self._week_cases_from_data(week_data))
        return np.asarray(self._estimation.quantile_model.predict(case_df, q=q), dtype=float)

    def compute_credibility(self, w: np.ndarray, week_data: WeekRecommendationData, realized: np.ndarray) -> float:
        d_post = self.compute_post_review(w=w, week_data=week_data)
        return float(np.mean(np.abs(np.asarray(realized, dtype=float) - d_post)))

    def _get_profile(self, surgeon_code: str):
        if self._estimation.response_profiler is not None:
            return self._estimation.response_profiler.get_profile(surgeon_code)
        return self._estimation.response_estimator.get_params(surgeon_code)

    def _normalize_residue(self, residue: float) -> float:
        residue = float(np.mod(residue, self._display_grid_step))
        if abs(residue - self._display_grid_step) <= 1e-8:
            return 0.0
        return residue

    def _make_display_grid(self, booking: float, L_bound: float, U_bound: float) -> np.ndarray:
        step = self._display_grid_step
        residue = self._display_grid_residue
        start = int(np.ceil((L_bound - residue) / step))
        end = int(np.floor((U_bound - residue) / step))
        if end < start:
            return np.asarray([float(np.clip(booking, L_bound, U_bound))], dtype=float)

        grid = residue + step * np.arange(start, end + 1, dtype=float)
        if not np.any(np.isclose(grid, booking, atol=1e-8, rtol=0.0)):
            grid = np.sort(np.unique(np.concatenate([grid, [float(booking)]])))
        return np.asarray(grid, dtype=float)

    def _display_cell_bounds(
        self,
        grid_values: np.ndarray,
        grid_index: int,
        L_bound: float,
        U_bound: float,
    ) -> Tuple[float, float]:
        if grid_index == 0:
            lower = float(L_bound)
        else:
            lower = 0.5 * (float(grid_values[grid_index - 1]) + float(grid_values[grid_index]))
        if grid_index == len(grid_values) - 1:
            upper = float(U_bound)
        else:
            upper = 0.5 * (float(grid_values[grid_index]) + float(grid_values[grid_index + 1]))
        return lower, upper

    def _delta_rec_interval_for_display_cell(
        self,
        booking: float,
        d_low: float,
        d_high: float,
        a: float,
        h_plus: float,
        h_minus: float,
        global_lb: float,
        global_ub: float,
        tol: float = 1e-8,
    ) -> Tuple[float, float] | None:
        if a <= tol:
            if d_low - tol <= booking <= d_high + tol:
                return global_lb, global_ub
            return None

        if d_high < booking - tol:
            lb = (d_low - booking) / a - h_minus
            ub = (d_high - booking) / a - h_minus
        elif d_low > booking + tol:
            lb = (d_low - booking) / a + h_plus
            ub = (d_high - booking) / a + h_plus
        else:
            lb = -h_minus if d_low >= booking - tol else (d_low - booking) / a - h_minus
            ub = h_plus if d_high <= booking + tol else (d_high - booking) / a + h_plus

        lb = max(global_lb, lb)
        ub = min(global_ub, ub)
        if ub + tol < lb:
            return None
        return float(lb), float(ub)

    def _nearest_grid_index(self, grid_values: np.ndarray, target: float, booking: float) -> int:
        best_idx = 0
        best_key = (float("inf"), float("inf"), float("inf"))
        for idx, g in enumerate(np.asarray(grid_values, dtype=float)):
            key = (abs(float(g) - target), abs(float(g) - booking), abs(float(g)))
            if key < best_key:
                best_key = key
                best_idx = idx
        return best_idx

    def _cases_to_quantile_df(self, cases: Sequence[CaseRecord]) -> pd.DataFrame:
        rows = []
        for c in cases:
            rows.append(
                {
                    Col.PROCEDURE_ID: c.procedure_id,
                    Col.PROCEDURE_DURATION: c.actual_duration_min,
                    Col.SURGEON_CODE: c.surgeon_code,
                    Col.CASE_SERVICE: c.service,
                    Col.MONTH: c.month,
                    Col.BOOKED_MINUTES: c.booked_duration_min,
                }
            )
        return pd.DataFrame(rows)

    def _week_cases_from_data(self, week_data: WeekRecommendationData) -> List[CaseRecord]:
        cases: List[CaseRecord] = []
        for i in range(week_data.n_cases):
            cases.append(
                CaseRecord(
                    case_id=i,
                    procedure_id="UNK",
                    surgeon_code=week_data.surgeon_codes[i],
                    service="Other",
                    patient_type="Elective",
                    operating_room="",
                    booked_duration_min=float(week_data.bookings[i]),
                    actual_duration_min=float(week_data.realized[i]),
                    actual_start=pd.Timestamp(date(2020, 1, 1)).to_pydatetime(),
                    week_of_year=1,
                    month=1,
                    year=2020,
                )
            )
        return cases

    def _df_to_cases(self, df: pd.DataFrame) -> List[CaseRecord]:
        defaults = {Col.MONTH: 1, Col.YEAR: 2020, Col.WEEK_OF_YEAR: 1}
        work = df.copy()
        for k, v in defaults.items():
            if k not in work.columns:
                work[k] = v
        if Col.ACTUAL_START not in work.columns:
            work[Col.ACTUAL_START] = pd.Timestamp(date(2020, 1, 1))

        cases: List[CaseRecord] = []
        for i, row in work.reset_index(drop=True).iterrows():
            cases.append(
                CaseRecord(
                    case_id=int(row.get(Col.CASE_UID, i)),
                    procedure_id=str(row.get(Col.PROCEDURE_ID, "UNK")),
                    surgeon_code=str(row.get(Col.SURGEON_CODE, "Other")),
                    service=str(row.get(Col.CASE_SERVICE, "Other")),
                    patient_type=str(row.get(Col.PATIENT_TYPE, "Elective")),
                    operating_room=str(row.get(Col.OPERATING_ROOM, "")),
                    booked_duration_min=float(row.get(Col.BOOKED_MINUTES, 0.0)),
                    actual_duration_min=float(row.get(Col.PROCEDURE_DURATION, row.get(Col.BOOKED_MINUTES, 0.0))),
                    actual_start=pd.to_datetime(row.get(Col.ACTUAL_START)).to_pydatetime(),
                    week_of_year=int(row.get(Col.WEEK_OF_YEAR, 1)),
                    month=int(row.get(Col.MONTH, 1)),
                    year=int(row.get(Col.YEAR, 2020)),
                    site=str(row.get(Col.SITE, "")),
                    surgical_duration_min=float(row.get(Col.SURGICAL_DURATION, 0.0)),
                )
            )
        return cases
