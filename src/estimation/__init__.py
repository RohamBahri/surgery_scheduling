"""Shared estimation package types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bootstrap import BootstrapEstimator
    from .inverse import CriticalRatioEstimator
    from .quantile import QuantileModel
    from .response import ResponseEstimator, ResponseProfiler


@dataclass
class EstimationResult:
    """Container for estimation artifacts shared across phases."""

    quantile_model: "QuantileModel | Any"
    critical_ratios: "CriticalRatioEstimator | Any"
    response_estimator: "ResponseEstimator | Any"
    response_profiler: "ResponseProfiler | Any"
    bootstrap: "BootstrapEstimator | Any | None" = None
