"""
Utilities for stochastic aspects of surgery scheduling, primarily for
Sample Average Approximation (SAA). Includes functions for building empirical
duration distributions and sampling scenarios.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.constants import (
    DataColumns,
    DomainConstants,
    LoggingConstants,
)

# Setup logger
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


def build_empirical_distributions(
    df_warm_up_data: pd.DataFrame, config: AppConfig
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Constructs empirical duration distributions from warm-up data.

    Creates a dictionary mapping each unique (recoded) 'main_procedure_id'
    to an array of its historical 'procedure_duration_min' values.
    Also returns an array of all pooled durations. Durations are clipped
    to be at least `MIN_PROCEDURE_DURATION`.

    Args:
        df_warm_up_data: DataFrame of warm-up period surgeries. Must contain
            `DataColumns.PROCEDURE_DURATION_MIN` and `DataColumns.MAIN_PROCEDURE_ID`.
        config: Application configuration (currently unused but kept for API consistency).

    Returns:
        A tuple containing:
        - procedure_specific_samples (Dict[str, np.ndarray]):
            Maps procedure_id to an array of historical durations.
        - all_pooled_samples (np.ndarray):
            Array of all valid historical durations pooled together.
    """
    if df_warm_up_data.empty:
        logger.warning(
            "Warm-up data is empty. Cannot build empirical distributions. "
            "Returning empty structures."
        )
        return {}, np.empty(0, dtype=float)

    if DataColumns.PROCEDURE_DURATION_MIN not in df_warm_up_data.columns:
        logger.error(
            f"'{DataColumns.PROCEDURE_DURATION_MIN}' not found in warm-up data. "
            "Cannot build empirical distributions."
        )
        return {}, np.empty(0, dtype=float)
    if DataColumns.MAIN_PROCEDURE_ID not in df_warm_up_data.columns:
        logger.error(
            f"'{DataColumns.MAIN_PROCEDURE_ID}' not found in warm-up data. "
            "Cannot build procedure-specific empirical distributions."
        )
        return {}, np.empty(0, dtype=float)

    # Filter for positive durations and clip to a minimum value
    df_valid_durations = df_warm_up_data[
        df_warm_up_data[DataColumns.PROCEDURE_DURATION_MIN] > 0
    ].copy()

    if df_valid_durations.empty:
        logger.warning(
            "No positive procedure durations found in warm-up data. Returning empty distributions."
        )
        return {}, np.empty(0, dtype=float)

    df_valid_durations[DataColumns.PROCEDURE_DURATION_MIN] = df_valid_durations[
        DataColumns.PROCEDURE_DURATION_MIN
    ].clip(lower=DomainConstants.MIN_PROCEDURE_DURATION)

    # Group by recoded procedure ID to get procedure-specific samples
    procedure_specific_samples: Dict[str, np.ndarray] = (
        df_valid_durations.groupby(DataColumns.MAIN_PROCEDURE_ID)[
            DataColumns.PROCEDURE_DURATION_MIN
        ]
        .apply(lambda durations_series: durations_series.to_numpy(dtype=float))
        .to_dict()
    )

    all_pooled_samples: np.ndarray = df_valid_durations[
        DataColumns.PROCEDURE_DURATION_MIN
    ].to_numpy(dtype=float)

    logger.info(
        f"Built empirical distributions for {len(procedure_specific_samples)} procedures. "
        f"Total pooled samples: {len(all_pooled_samples)}."
    )
    return procedure_specific_samples, all_pooled_samples


def sample_scenarios(
    surgeries_for_sampling: List[Dict[str, Any]],
    procedure_specific_samples: Dict[str, np.ndarray],
    all_pooled_samples: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    """Generates a matrix of sampled surgery durations for SAA.

    For each surgery in `surgeries_for_sampling`, K scenarios are generated.
    The sampling hierarchy is:
    1. If the surgery's 'proc_id' has specific samples, draw from them.
    2. Else, if `all_pooled_samples` is non-empty, draw from it.
    3. Else (no samples available), use the surgery's 'booked_min' as a
       deterministic fallback for all scenarios.

    Args:
        surgeries_for_sampling: List of surgery dictionaries. Each must contain
            'proc_id' and 'booked_min'.
        procedure_specific_samples: Dictionary mapping procedure_id to its
            empirical duration samples.
        all_pooled_samples: Array of all pooled empirical durations.
        config: Application configuration containing SAA settings.

    Returns:
        A NumPy array of shape (num_surgeries, num_saa_scenarios) containing
        sampled durations.
    """
    num_surgeries = len(surgeries_for_sampling)
    num_scenarios_k = config.saa.scenarios

    if num_surgeries == 0:
        logger.info(
            "No surgeries provided for scenario sampling. Returning empty matrix."
        )
        return np.zeros((0, num_scenarios_k), dtype=float)

    # Initialize Random Number Generator with a seed for reproducibility
    random_seed = config.saa.random_seed
    rng = np.random.default_rng(random_seed)
    logger.debug(f"Scenario sampling RNG initialized with seed: {random_seed}")

    scenario_matrix = np.zeros((num_surgeries, num_scenarios_k), dtype=float)

    for i, surgery_info in enumerate(surgeries_for_sampling):
        procedure_id = surgery_info.get("proc_id")
        samples_for_this_surgery: Optional[np.ndarray] = None

        if procedure_id and procedure_id in procedure_specific_samples:
            samples_for_this_surgery = procedure_specific_samples[procedure_id]
            if (
                samples_for_this_surgery is not None
                and len(samples_for_this_surgery) == 0
            ):
                samples_for_this_surgery = None  # Treat empty array as no samples

        if samples_for_this_surgery is None and len(all_pooled_samples) > 0:
            samples_for_this_surgery = all_pooled_samples
            logger.debug(
                f"Surgery {i} (ProcID: {procedure_id}): No specific samples found. "
                "Using pooled samples."
            )

        if samples_for_this_surgery is not None and len(samples_for_this_surgery) > 0:
            scenario_matrix[i, :] = rng.choice(
                samples_for_this_surgery, size=num_scenarios_k, replace=True
            )
        else:
            # Fallback to booked time, ensuring it's at least MIN_PROCEDURE_DURATION
            booked_duration = surgery_info.get(DataColumns.BOOKED_MIN)
            if booked_duration is None:
                logger.warning(
                    f"Surgery {i} (ProcID: {procedure_id}): No samples and no '{DataColumns.BOOKED_MIN}'. "
                    f"Falling back to {DomainConstants.MIN_PROCEDURE_DURATION} for scenarios."
                )
                fallback_duration = DomainConstants.MIN_PROCEDURE_DURATION
            else:
                fallback_duration = max(
                    DomainConstants.MIN_PROCEDURE_DURATION, float(booked_duration)
                )

            scenario_matrix[i, :] = fallback_duration
            logger.debug(
                f"Surgery {i} (ProcID: {procedure_id}): No specific or pooled samples. "
                f"Using fallback duration {fallback_duration} for all scenarios."
            )

    logger.info(f"Generated scenario matrix of shape {scenario_matrix.shape} for SAA.")
    return scenario_matrix
