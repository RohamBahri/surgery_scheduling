"""Exact VFCG method implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import Col, ScheduleResult, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
from src.data.scope import apply_experiment_scope
from src.estimation import EstimationResult
from src.estimation.orchestrator import fit_estimation_pipeline
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.methods.base import Method
from src.planning.instance import build_weekly_instance
from src.solvers.deterministic import solve_deterministic
from src.vfcg.solver import vfcg_solve
from src.vfcg.types import CertificationResult, VFCGResult


class VFCGMethod(Method):
    def __init__(self, config: Config):
        super().__init__(name="VFCG", config=config)
        self._estimation_result: EstimationResult | None = None
        self._recommendation_model: RecommendationModel | None = None
        self._vfcg_result: VFCGResult | None = None
        self._training_weeks: list[int] = []
        self._training_summary: dict[str, object] = {}

    @property
    def training_summary(self) -> dict[str, object]:
        return dict(self._training_summary)

    def fit(self, df_train: pd.DataFrame) -> None:
        self._estimation_result = fit_estimation_pipeline(
            df_train=df_train,
            config=self.config,
            skip_profiles=False,
            quiet=True,
            skip_bootstrap=True,
            run_bootstrap=False,
        )

        self._recommendation_model = RecommendationModel(
            estimation_result=self._estimation_result,
            costs=self.config.costs,
            plausibility_tails=(self.config.vfcg.plausibility_tau_L, self.config.vfcg.plausibility_tau_U),
            w_max=self.config.vfcg.w_max,
        )

        # Explicit legacy-aligned training-week construction path.
        df_scoped, _ = apply_experiment_scope(df_train, self.config)
        candidate_pools = build_candidate_pools(df_scoped, self.config)
        eligibility_maps = build_eligibility_maps(df_train, self.config)

        dt = pd.to_datetime(df_scoped[Col.ACTUAL_START], errors="coerce")
        week_starts = sorted((dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize().dropna().unique())
        selected_week_pairs = list(enumerate(week_starts))
        if self.config.vfcg.max_training_weeks is not None:
            k = int(self.config.vfcg.max_training_weeks)
            selected_week_pairs = selected_week_pairs[-k:]

        selected_week_starts = [pd.Timestamp(ws) for _, ws in selected_week_pairs]
        if selected_week_starts:
            actual_norm = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()
            self._recommendation_model.prepare(df_scoped.loc[actual_norm.isin(selected_week_starts)].copy())
        else:
            self._recommendation_model.prepare(df_scoped.iloc[0:0].copy())

        week_data_list: list[WeekRecommendationData] = []
        week_indices: list[int] = []
        for w_idx, ws in selected_week_pairs:
            instance = build_weekly_instance(
                df_pool=df_scoped,
                horizon_start=pd.Timestamp(ws),
                week_index=w_idx,
                config=self.config,
                candidate_pools=candidate_pools,
                eligibility_maps=eligibility_maps,
            )
            if instance.num_cases == 0:
                continue
            week_data_list.append(self._recommendation_model.prepare_instance(instance))
            week_indices.append(w_idx)

        self._training_weeks = week_indices

        if not week_data_list:
            w0 = np.zeros(self._recommendation_model.feature_dim, dtype=float)
            self._vfcg_result = VFCGResult(
                w_optimal=w0,
                objective=float("nan"),
                realized_objective=float("nan"),
                credibility_mae=float("nan"),
                credibility_slack=float("nan"),
                n_iterations=0,
                certification=CertificationResult(
                    status="NO_TRAINING_WEEKS",
                    max_violation=0.0,
                    reconstructed_objective=0.0,
                    reconstructed_realized_objective=0.0,
                    reconstructed_credibility_mae=0.0,
                    reconstructed_credibility_slack=0.0,
                    master_objective=0.0,
                    master_realized_objective=0.0,
                    master_bound=0.0,
                    tie_break_flags=None,
                ),
                iterations=[],
                total_cuts_added=0,
            )
            self._training_summary = {
                "selected_training_weeks": [],
                "learned_weights": w0.tolist(),
                "feature_dim": int(self._recommendation_model.feature_dim),
                "feature_names": list(self._recommendation_model.feature_names),
                "master_objective": None,
                "master_bound": None,
                "training_objective": None,
                "training_bound": None,
                "training_gap": None,
                "certification_status": "NO_TRAINING_WEEKS",
                "vfcg_iterations": 0,
                "vfcg_total_cuts": 0,
            }
            return

        self._vfcg_result = vfcg_solve(
            week_data_list=week_data_list,
            recommendation_model=self._recommendation_model,
            config=self.config,
            costs=self.config.costs,
            capacity_cfg=self.config.capacity,
            solver_cfg=self.config.solver,
            turnover=self.config.capacity.turnover_minutes,
        )

        self._training_summary = {
            "selected_training_weeks": week_indices,
            "learned_weights": self._vfcg_result.w_optimal.tolist(),
            "feature_dim": int(self._recommendation_model.feature_dim),
            "feature_names": list(self._recommendation_model.feature_names),
            "master_objective": float(self._vfcg_result.objective),
            "master_realized_objective": float(self._vfcg_result.realized_objective),
            "master_bound": float(self._vfcg_result.certification.master_bound),
            "training_objective": float(self._vfcg_result.realized_objective),
            "training_penalized_objective": float(self._vfcg_result.objective),
            "training_bound": float(self._vfcg_result.certification.master_bound),
            "training_gap": float(self._vfcg_result.iterations[-1].master_gap) if self._vfcg_result.iterations else None,
            "training_credibility_mae": float(self._vfcg_result.credibility_mae),
            "training_credibility_slack": float(self._vfcg_result.credibility_slack),
            "certification_status": self._vfcg_result.certification.status,
            "vfcg_iterations": int(self._vfcg_result.n_iterations),
            "vfcg_total_cuts": int(self._vfcg_result.total_cuts_added),
            "vfcg_max_violation": float(self._vfcg_result.certification.max_violation),
            "vfcg_tie_break_flags": self._vfcg_result.certification.tie_break_flags,
        }

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        if self._recommendation_model is None or self._vfcg_result is None:
            raise RuntimeError("VFCGMethod must be fitted before planning.")

        week_data = self._recommendation_model.prepare_instance(instance)
        d_post = self._recommendation_model.compute_post_review(self._vfcg_result.w_optimal, week_data)
        return solve_deterministic(
            cases=instance.cases,
            durations=np.asarray(d_post, dtype=float),
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            case_eligible_blocks=instance.case_eligible_blocks,
            turnover=self.config.capacity.turnover_minutes,
            model_name=self.name,
        )
