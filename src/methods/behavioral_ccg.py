from __future__ import annotations

import numpy as np
import pandas as pd

from src.bilevel.ccg import CCGResult, solve_bilevel_ccg
from src.core.config import Config
from src.core.types import Col, ScheduleResult, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
from src.data.scope import apply_experiment_scope
from src.estimation import EstimationResult
from src.estimation.orchestrator import fit_estimation_pipeline
from src.estimation.recommendation import RecommendationModel
from src.methods.base import Method
from src.planning.instance import build_weekly_instance
from src.solvers.deterministic import solve_deterministic


class BehavioralCCGMethod(Method):
    def __init__(self, config: Config):
        super().__init__(name="BehavioralCCG", config=config)
        self._estimation_result: EstimationResult | None = None
        self._recommendation_model: RecommendationModel | None = None
        self._ccg_result: CCGResult | None = None

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
            plausibility_tails=(
                self.config.bilevel.plausibility_tau_L,
                self.config.bilevel.plausibility_tau_U,
            ),
            w_max=self.config.bilevel.w_max,
        )
        self._recommendation_model.prepare(df_train)

        df_scoped, _ = apply_experiment_scope(df_train, self.config)
        candidate_pools = build_candidate_pools(df_scoped, self.config)
        eligibility_maps = build_eligibility_maps(df_train, self.config)

        dt = pd.to_datetime(df_scoped[Col.ACTUAL_START], errors="coerce")
        week_starts = sorted((dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize().dropna().unique())

        week_data_list = []
        for w_idx, ws in enumerate(week_starts):
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

        if not week_data_list:
            self._ccg_result = type("R", (), {"w_optimal": np.zeros(self._recommendation_model.feature_dim)})()
            return

        self._ccg_result = solve_bilevel_ccg(
            week_data_list=week_data_list,
            recommendation_model=self._recommendation_model,
            config=self.config.bilevel,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            turnover=self.config.capacity.turnover_minutes,
        )

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        if self._recommendation_model is None or self._ccg_result is None:
            raise RuntimeError("BehavioralCCGMethod must be fitted before planning.")

        week_data = self._recommendation_model.prepare_instance(instance)
        d_post = self._recommendation_model.compute_post_review(self._ccg_result.w_optimal, week_data)
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
