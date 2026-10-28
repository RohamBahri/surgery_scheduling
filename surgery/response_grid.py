"""Apply a hypothesized TRUE surgeon response to frozen displayed advice.

No historical adoption data are observed; these are scenario assumptions.
The trained policy stays fixed when the true-response configuration changes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from surgery.nonmonotone_learning import response,response_saturating


@dataclass(frozen=True)
class Scenario:
    name: str
    link: str
    alpha: float
    h: float


def implemented_from_display(booked,display,scenario):
    """Never re-canonicalize the frozen display under the new response."""
    if scenario.link not in ("saturating","nonmonotone"):
        raise ValueError("Unknown true-response link")
    if not (0<scenario.alpha<1 and scenario.h>0):
        raise ValueError("Invalid hypothetical surgeon response")
    b=np.asarray(booked,float)
    u=np.clip(np.asarray(display,float),-180.,180.)
    if b.shape!=u.shape:
        raise ValueError("Duration and displayed-advice shapes must match")
    f=response_saturating if scenario.link=="saturating" else response
    return np.maximum(b+f(u,scenario.alpha,scenario.h),1.)
