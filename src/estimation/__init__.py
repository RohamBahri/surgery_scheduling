from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bootstrap import BootstrapResult
    from .inverse import CriticalRatioEstimator
    from .profiles import ResponseProfiler
    from .quantile_model import ConditionalQuantileModel
    from .response import ResponseEstimator


@dataclass
class EstimationResult:
    quantile_model: "ConditionalQuantileModel"
    critical_ratios: "CriticalRatioEstimator"
    response_estimator: "ResponseEstimator"
    response_profiler: "ResponseProfiler | None"
    bootstrap: "BootstrapResult | None" = None
