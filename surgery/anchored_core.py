"""Pure mathematical layer of anchored regret learning (no optimization solver).

The fixed-slope anchored upper bound is deliberately distinguished from the
schedule-specific tangent bound. Both bound primary-optimal regret, not the
realized cost of a tolerance-lexicographic planner without its tie correction.
"""
from dataclasses import dataclass
import numpy as np

from .planner import display_and_duration, reachable
from .weekly_planner import evaluate

GAMMA_DEFAULT = 2.0
OVER = 2.75
TURN = 30.0
CAP = 480.0


def key(component):
    return f"{component.group}|{component.monday}|{int(min(component.case_ids))}"


def phi(component, assignment, durations):
    """Primary scheduling score after removing duration-only constants."""
    from .weekly_planner import loads
    a = np.asarray(assignment, int)
    counts = np.bincount(a, minlength=component.k)
    return 30.0 * float((counts > 0).sum()) + OVER * float(
        np.maximum(loads(component, a, durations) - CAP, 0.).sum())


def cost_identity(component, assignment, durations):
    d = np.asarray(durations, float)
    return CAP * component.k - d.sum() - TURN * component.n + phi(component, assignment, d)


def slopes_from_reference(component, assignment, beta, tol=1e-8):
    from .weekly_planner import loads
    load = loads(component, assignment, beta)
    return np.where(load > CAP + tol, OVER, 0.)


def fixed_slope_lower(component, assignment, d, slopes):
    """Global affine lower bound on phi(z,d); slopes fixed by physical room.

    For an unused room the expression is exactly zero, not -480*slope.
    """
    a = np.asarray(assignment, int)
    d = np.asarray(d, float)
    slope = np.asarray(slopes, float)
    if slope.shape != (component.k,) or np.any(slope < 0) or np.any(slope > OVER):
        raise ValueError("Every physical-room slope must lie in [0,2.75]")
    counts = np.bincount(a, minlength=component.k)
    used = counts > 0
    load = np.bincount(a, weights=d + TURN, minlength=component.k) - TURN * used
    return float(TURN * used.sum() + np.dot(slope, load - CAP * used))


def exact_tangent(component, assignment, beta, d):
    """Schedule-specific supporting affine function at beta."""
    slope = slopes_from_reference(component, assignment, beta)
    return fixed_slope_lower(component, assignment, d, slope)


def anchored_cut(component, anchor, assignment, d, gamma=GAMMA_DEFAULT,
                 exact=False):
    """Evaluate a valid reference cut; a dict may specify a kink subgradient.

    Exact-separation cuts carry physical-room slopes selected at the beta
    reference point. At 480 exactly, both 0 and OVER are valid tangents.
    Preserving this choice is essential for master/oracle consistency.
    """
    if isinstance(assignment, dict):
        slopes = np.asarray(assignment['slopes'], float)
        assignment = assignment['assignment']
        supporting = fixed_slope_lower(component, assignment, d, slopes)
    elif exact:
        supporting = exact_tangent(component, assignment, anchor['beta'], d)
    else:
        supporting = fixed_slope_lower(component, assignment, d, anchor['slopes'])
    return float(evaluate(component, assignment, component.actual)['cost'] - gamma * supporting)


def anchored_score(component, anchor, competitor_assignments, d, oracle_cost,
                   gamma=GAMMA_DEFAULT, exact=False):
    if not competitor_assignments:
        raise ValueError("At least one competitor is required")
    return (gamma * phi(component, anchor['assignment'], d)
            + max(anchored_cut(component, anchor, a, d, gamma, exact)
                  for a in competitor_assignments)
            - oracle_cost)


def reference_fingerprint(beta):
    """Stable identity of the duration point defining the tangent regimes."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(beta,dtype=np.float64)).tobytes()).hexdigest()


def normalize_tangent(component, item, beta):
    """Return an exact supporting cut of phi(z;d) at the current beta.

    If a previously separated kink slope was generated for another anchor,
    it is recomputed at the new beta rather than falsely called a tangent.
    """
    if isinstance(item, dict):
        z=list(map(int,item['assignment']))
        match=item.get('anchor_sha256')==reference_fingerprint(beta)
        if match and 'slopes' in item:
            slopes=np.asarray(item['slopes'],float)
        else:
            slopes=slopes_from_reference(component,z,beta)
    else:
        z=list(map(int,item))
        slopes=slopes_from_reference(component,z,beta)
    if len(z)!=component.n or slopes.shape!=(component.k,):
        raise ValueError('Invalid scheduling cut shape')
    if np.any(~np.isclose(slopes,0.,atol=1e-8)&
              ~np.isclose(slopes,OVER,atol=1e-8)):
        raise ValueError('Tangent has invalid slope values')
    from .weekly_planner import loads
    roomloads=loads(component,z,beta)
    if np.any((slopes>OVER/2)&(roomloads<CAP-1e-6)):
        raise ValueError('Overtime tangent inconsistent with beta')
    if np.any((slopes<OVER/2)&(roomloads>CAP+1e-6)):
        raise ValueError('Idle tangent inconsistent with beta')
    return {'assignment':z,'slopes':slopes.tolist(),
            'anchor_sha256':reference_fingerprint(beta)}


def tangent_key(cut):
    return (tuple(cut['assignment']),
            tuple(round(float(v),7) for v in cut['slopes']))


def response_single(score, booked, alpha, h):
    _, d = display_and_duration(np.array([score]), np.array([booked]), alpha, h)
    return float(d[0] - booked)


@dataclass(frozen=True)
class HingeDC:
    """Exact scalar correction r(s)=constant+P(s)-Q(s), P,Q convex PWL."""
    constant: float
    positive: tuple  # (kink, positive coefficient)
    negative: tuple  # (kink, positive coefficient)

    def parts(self, scores):
        s = np.asarray(scores, float)
        p = np.zeros_like(s)
        q = np.zeros_like(s)
        for a, v in self.positive:
            p += v * np.maximum(s - a, 0.)
        for a, v in self.negative:
            q += v * np.maximum(s - a, 0.)
        return p, q

    def affine_tangents(self, at):
        """Intercept+slope*s at a chosen point; right derivative at kinks."""
        a = float(at)
        ps = sum(v for k, v in self.positive if a >= k)
        qs = sum(v for k, v in self.negative if a >= k)
        p, q = self.parts(a)
        return (float(p) - ps * a, float(ps), float(q) - qs * a, float(qs))

    def correction(self, scores):
        p, q = self.parts(scores)
        return self.constant + p - q

    def duration_bounds(self, scores, at):
        """Touching DC lower/upper bounds on corrected duration."""
        p, q = self.parts(scores)
        pc, ps, qc, qs = self.affine_tangents(at)
        return self.constant + pc + ps * scores - q, self.constant + p - qc - qs * scores


def hinge_response(booked, alpha=.8, h=30.):
    """Construct exact hinge decomposition, including display clipping.

    Uses finite differences of an explicitly known PWL function only to
    extract the slopes; the result is algebraic to floating-point precision.
    """
    if not (0 < alpha <= 1) or h <= 0 or booked <= 0:
        raise ValueError("Invalid response parameters or booking")
    lo, hi = max(-180., 1. - booked), 180.
    q = 1. - alpha
    knots = [lo, hi, -h, h]
    if q > 0:
        knots += [-h / q, h / q]
    knots = sorted(set(float(t) for t in knots if lo <= t <= hi))
    def r(v):
        return response_single(v, booked, alpha, h)
    base = r(knots[0] - 1.)
    past_slope = 0.
    pos, neg = [], []
    for j, knot in enumerate(knots):
        right = knots[j+1] if j+1 < len(knots) else knot + 1.
        # Just to the right of the current kink, safely within the segment.
        s1 = knot + 0.25 * (right - knot)
        s2 = knot + 0.75 * (right - knot)
        slope = (r(s2) - r(s1)) / (s2 - s1)
        change = slope - past_slope
        if change > 1e-10:
            pos.append((knot, change))
        elif change < -1e-10:
            neg.append((knot, -change))
        past_slope = slope
    return HingeDC(base, tuple(pos), tuple(neg))


def response_decomposition(booked, alpha=.8, h=30.):
    return [hinge_response(float(b), alpha, h) for b in np.asarray(booked)]


def bound_checks(component, anchor, d, oracle_value, competitors, gamma=2.,
                 realized=None):
    """Finite-library value is only a lower bound on the complete anchored S.

    With an exactly primary-optimal deployment, full anchored S bounds regret;
    this function checks the identity on a supplied subset without presenting
    it as a full certificate.
    """
    for a in competitors:
        if fixed_slope_lower(component, a, d, anchor['slopes']) > phi(component, a, d) + 1e-6:
            raise AssertionError("Invalid global affine room underestimator")
        if exact_tangent(component, a, anchor['beta'], anchor['beta']) > phi(component, a, anchor['beta']) + 1e-6:
            raise AssertionError("Schedule-specific tangent not touching")
    return anchored_score(component, anchor, competitors, d, oracle_value, gamma)
