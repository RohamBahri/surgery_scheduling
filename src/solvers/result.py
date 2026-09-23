"""Backend-independent result data; preserve the solver's native objective/bound."""

from dataclasses import asdict, dataclass
from math import inf, isfinite

from src.core.column import ScheduleColumn


def status_name(status: int) -> str:
    # Gurobi's documented status codes, kept here without importing a backend.
    names = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    return names.get(status, str(status))


def absolute_gap(ub: float | None, lb: float | None) -> float:
    return max(0.0, ub - lb) if ub is not None and lb is not None else inf


def relative_gap(ub: float | None, lb: float | None) -> float:
    gap = absolute_gap(ub, lb)
    if gap == 0:
        return 0.0
    return gap / abs(ub) if ub and isfinite(ub) else inf


@dataclass(frozen=True)
class SolveDiagnostics:
    status: str
    obj_val: float | None
    obj_bound: float | None
    absolute_gap: float
    relative_gap: float
    runtime_seconds: float
    sol_count: int
    proven_optimal: bool
    status_code: int | None = None
    backend: str = "gurobi"

    def as_dict(self) -> dict:
        return asdict(self)


def diagnostics_from_model(model) -> SolveDiagnostics:
    count = int(model.SolCount)
    ub = float(model.ObjVal) if count else None
    try:
        lb = float(model.ObjBound)
    except (AttributeError, RuntimeError):
        lb = None
    # Some early termination states do not expose a bound attribute.
    status = int(model.Status)
    return SolveDiagnostics(
        status_name(status),
        ub,
        lb,
        absolute_gap(ub, lb),
        relative_gap(ub, lb),
        float(model.Runtime),
        count,
        status == 2,
        status,
    )


@dataclass(frozen=True)
class PricingResult:
    column: ScheduleColumn | None
    diagnostics: SolveDiagnostics
