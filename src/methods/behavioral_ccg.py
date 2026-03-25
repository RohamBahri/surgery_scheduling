from __future__ import annotations

import numpy as np
import pandas as pd

from src.bilevel.ccg import CCGResult, solve_bilevel_ccg
from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
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

        start = pd.to_datetime(df_train["actual_start"]).min().normalize()
        candidate_pools = build_candidate_pools(df_train, self.config)
        eligibility_maps = build_eligibility_maps(df_train, self.config)
        train_instance = build_weekly_instance(
            df_pool=df_train,
            horizon_start=start,
            week_index=0,
            config=self.config,
            candidate_pools=candidate_pools,
            eligibility_maps=eligibility_maps,
        )

        week_data = self._recommendation_model.prepare_instance(train_instance)
        self._ccg_result = solve_bilevel_ccg(
            week_data_list=[week_data],
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
