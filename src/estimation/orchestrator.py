"""End-to-end estimation pipeline orchestration."""

from __future__ import annotations

import logging

import pandas as pd

from src.core.config import Config
from src.estimation import EstimationResult
from src.estimation.inverse import CriticalRatioEstimator
from src.estimation.profiles import ResponseProfiler
from src.estimation.quantile_model import ConditionalQuantileModel
from src.estimation.response import ResponseEstimator

logger = logging.getLogger(__name__)


def fit_estimation_pipeline(
    df_train: pd.DataFrame,
    config: Config,
    skip_profiles: bool = False,
    quiet: bool = False,
    skip_bootstrap: bool = True,
) -> EstimationResult:
    if not quiet:
        logger.info("Fitting conditional quantile model...")
    quantile_model = ConditionalQuantileModel(config.estimation.quantile_model).fit(df_train)

    if not quiet:
        logger.info("Estimating surgeon critical ratios...")
    critical_ratios = CriticalRatioEstimator(quantile_model, config.estimation.inverse).fit(df_train)

    if not quiet:
        logger.info("Estimating surgeon response parameters...")
    response_estimator = ResponseEstimator(
        quantile_model,
        critical_ratios,
        config.estimation.response,
    ).fit(df_train)

    response_profiler = None
    if not skip_profiles:
        if not quiet:
            logger.info("Fitting response profiles...")
        surgeon_services = (
            df_train.groupby("surgeon_code")["case_service"]
            .agg(lambda x: x.mode().iat[0])
            .to_dict()
        )
        params_df = response_estimator.get_all_params()
        response_profiler = ResponseProfiler(config.estimation.profile).fit(params_df, surgeon_services)

    bootstrap = None
    if not skip_bootstrap and not quiet:
        logger.info("Bootstrap requested but disabled in this phase.")

    return EstimationResult(
        quantile_model=quantile_model,
        critical_ratios=critical_ratios,
        response_estimator=response_estimator,
        response_profiler=response_profiler,
        bootstrap=bootstrap,
    )
