"""Frozen-advice response scenario grid for UHN recommendation experiments.

A policy is trained under exactly one behavioral response assumption.
It produces a displayed advice amount, once, using the TRAINING link.
A deployed response scenario processes that SAME displayed advice:
re-canonicalizing advice under a new scenario would erase mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from surgery.nonmonotone_learning import (
    ResponseDC, response, response_saturating)

ALPHA_TRAIN=.8
H=30.
DISPLAY_CAP=180.
DURATION_FLOOR=1.

@dataclass(frozen=True)
class Scenario:
    name: str
    link: str
    alpha: float
    h: float = H

# All scenarios have the monotone saturating B response link.
# Training uses the two adoption rates and two meaningful display ranges.
TRAIN_SCENARIOS=(
    Scenario("SAT_A50_H30","saturating",.5,30.),
    Scenario("SAT_A50_H60","saturating",.5,60.),
    Scenario("SAT_A80_H30","saturating",.8,30.),
    Scenario("SAT_A80_H60","saturating",.8,60.),
)
# Evaluation also varies a narrower h=20 threshold, without retraining.
EVAL_SCENARIOS=(
    Scenario("SAT_A50_H20","saturating",.5,20.),
    Scenario("SAT_A50_H30","saturating",.5,30.),
    Scenario("SAT_A50_H60","saturating",.5,60.),
    Scenario("SAT_A80_H20","saturating",.8,20.),
    Scenario("SAT_A80_H30","saturating",.8,30.),
    Scenario("SAT_A80_H60","saturating",.8,60.),
)


def mapping(booked, scenario):
    """Exact compiler for policy FITTING, not the evaluation-time response."""
    return ResponseDC(booked,scenario.alpha,scenario.h,DISPLAY_CAP,
                      DURATION_FLOOR,link=scenario.link)


def advice_for_policy(raw, booked, trained_scenario):
    """Canonical display is determined by TRAINED policy only."""
    model=mapping(booked,trained_scenario)
    return np.asarray(model.display(raw),float)


def implemented_from_display(booked,display,scenario):
    """Apply the evaluation scenario to a FROZEN displayed correction."""
    b=np.asarray(booked,float)
    u=np.clip(np.asarray(display,float),-DISPLAY_CAP,DISPLAY_CAP)
    if b.shape!=u.shape:
        raise ValueError("Duration and displayed-advice shape mismatch")
    fn=response_saturating if scenario.link=="saturating" else response
    effective=fn(u,scenario.alpha,scenario.h)
    return np.maximum(b+effective,DURATION_FLOOR)


def naive_display(X,weights):
    """Response-unaware duration-error prediction shown literally as advice."""
    return np.clip(np.asarray(X,float)@np.asarray(weights,float),
                   -DISPLAY_CAP,DISPLAY_CAP)


def verify_same_scenario(raw,booked,scenario):
    """Training link and displayed deployment must agree on reviewed time."""
    model=mapping(booked,scenario)
    model.verify(raw)
    shown=advice_for_policy(raw,booked,scenario)
    adopted=implemented_from_display(booked,shown,scenario)
    if not np.allclose(adopted,model.reviewed(raw),atol=1e-8,rtol=0):
        raise AssertionError("Training/deployment response mismatch")
    return shown


def evaluation_policy_labels():
    """A distinct CASE and VF policy under each of four training scenarios."""
    result=[]
    for train in TRAIN_SCENARIOS:
        result.extend((f"CASE__{train.name}",f"VF__{train.name}"))
    return tuple(result)


def check_grid():
    assert len(TRAIN_SCENARIOS)==4 and len(EVAL_SCENARIOS)==6
    assert len({s.name for s in EVAL_SCENARIOS})==6
    assert set(TRAIN_SCENARIOS).issubset(set(EVAL_SCENARIOS))
    assert set(s.link for s in (*TRAIN_SCENARIOS,*EVAL_SCENARIOS))=={"saturating"}
    for s in EVAL_SCENARIOS:
        if not (0<s.alpha<1 and s.h>0):
            raise ValueError("Invalid scenario")
    return True
