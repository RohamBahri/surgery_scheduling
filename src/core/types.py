"""Canonical data types shared across the entire project.

These types form the interface contract between data loading, methods,
solvers, and evaluation.  Every module communicates through these objects
rather than through raw DataFrames or dictionaries, which prevents silent
schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ─── Column name constants ───────────────────────────────────────────────────

class Col:
    """Canonical DataFrame column names."""
    PATIENT_ID          = "patient_id"
    PATIENT_TYPE        = "patient_type"
    CASE_SERVICE        = "case_service"
    MAIN_PROCEDURE      = "main_procedure"
    PROCEDURE_ID        = "main_procedure_id"
    OPERATING_ROOM      = "operating_room"
    SURGEON             = "surgeon"
    SURGEON_CODE        = "surgeon_code"
    BOOKED_MINUTES      = "booked_time_minutes"
    ACTUAL_START        = "actual_start"
    ACTUAL_STOP         = "actual_stop"
    ENTER_ROOM          = "enter_room"
    LEAVE_ROOM          = "leave_room"
    PROCEDURE_DURATION  = "procedure_duration_min"
    PREPARATION_DURATION = "preparation_duration_min"
    WEEK_OF_YEAR        = "week_of_year"
    MONTH               = "month"
    YEAR                = "year"

    # Original Excel column pairs (date + time) — used only during loading
    ACTUAL_START_DATE = "actual_start_date"
    ACTUAL_START_TIME = "actual_start_time"
    ACTUAL_STOP_DATE  = "actual_stop_date"
    ACTUAL_STOP_TIME  = "actual_stop_time"
    ENTER_ROOM_DATE   = "enter_room_date"
    ENTER_ROOM_TIME   = "enter_room_time"
    LEAVE_ROOM_DATE   = "leave_room_date"
    LEAVE_ROOM_TIME   = "leave_room_time"


# ─── Domain constants ────────────────────────────────────────────────────────

class Domain:
    """Numerical and categorical constants."""
    MIN_PROCEDURE_DURATION = 30        # minutes
    MAX_OVERTIME_PER_BLOCK = 240       # minutes
    OR_ROOM_PREFIX = "OR"
    EMERGENCY_PATIENT = "EMERGENCY PATIENT"
    OTHER = "Other"
    UNKNOWN = "Unknown"
    EMERGENCY_ROOMS = ("OREMER", "ORER")


# ─── Record types ────────────────────────────────────────────────────────────

@dataclass
class CaseRecord:
    """One elective surgery case.

    Methods receive these in a ``WeeklyInstance``.  The contract is that a
    method's ``plan()`` must NOT read ``actual_duration_min`` — that field
    exists solely for evaluation after scheduling.
    """
    case_id: int
    procedure_id: str
    surgeon_code: str
    service: str
    patient_type: str
    operating_room: str
    booked_duration_min: float
    actual_duration_min: float      # evaluation only — do not use in plan()
    actual_start: datetime
    week_of_year: int
    month: int
    year: int


# ─── Block types ─────────────────────────────────────────────────────────────

# A block is identified by a (day_index, block_index) tuple throughout.
BlockId = Tuple[int, int]


@dataclass(frozen=True)
class CandidateBlock:
    """One OR block that *could* be opened in a planning week.

    The planner decides which candidates to open (staff) at activation
    cost ``activation_cost``.  Cases can only be assigned to opened blocks.
    """
    day_index: int
    block_index: int
    room: str
    capacity_minutes: float
    activation_cost: float

    @property
    def id(self) -> BlockId:
        return (self.day_index, self.block_index)


@dataclass
class BlockCalendar:
    """Candidate OR blocks for one planning week.

    ``candidates`` lists every block that *could* be opened.  The solver
    decides which ones to actually staff.  This replaces the old design
    where ``daily_counts`` specified a fixed number of guaranteed blocks.
    """
    candidates: List[CandidateBlock]

    @property
    def block_ids(self) -> List[BlockId]:
        """Flat list of (day_index, block_index) identifiers."""
        return [c.id for c in self.candidates]

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    def capacity(self, block_id: BlockId) -> float:
        """Nominal capacity of a specific candidate block."""
        for c in self.candidates:
            if c.id == block_id:
                return c.capacity_minutes
        raise KeyError(f"Block {block_id} not in calendar.")

    def activation_cost(self, block_id: BlockId) -> float:
        """Activation cost of a specific candidate block."""
        for c in self.candidates:
            if c.id == block_id:
                return c.activation_cost
        raise KeyError(f"Block {block_id} not in calendar.")

    def blocks_on_day(self, day: int) -> List[CandidateBlock]:
        return [c for c in self.candidates if c.day_index == day]


# ─── Instance type ───────────────────────────────────────────────────────────

@dataclass
class WeeklyInstance:
    """Everything needed to plan one week and evaluate the result.

    The ``cases`` list is the candidate set.  A method must assign each
    case to an opened block or defer it.
    """
    week_index: int
    start_date: date
    end_date: date
    cases: List[CaseRecord]
    calendar: BlockCalendar

    @property
    def num_cases(self) -> int:
        return len(self.cases)

    def booked_durations(self) -> List[float]:
        return [c.booked_duration_min for c in self.cases]

    def actual_durations(self) -> List[float]:
        return [c.actual_duration_min for c in self.cases]


# ─── Schedule types ──────────────────────────────────────────────────────────

@dataclass
class ScheduleAssignment:
    """One case's scheduling decision."""
    case_id: int
    day_index: Optional[int] = None     # None → deferred
    block_index: Optional[int] = None   # None → deferred

    @property
    def is_deferred(self) -> bool:
        return self.day_index is None

    @property
    def block_id(self) -> Optional[BlockId]:
        if self.is_deferred:
            return None
        return (self.day_index, self.block_index)


@dataclass
class ScheduleResult:
    """Complete output of a method's ``plan()`` call."""
    assignments: List[ScheduleAssignment]
    opened_blocks: Set[BlockId] = field(default_factory=set)
    solver_status: str = "Unknown"
    objective_value: Optional[float] = None
    solve_time_seconds: float = 0.0

    def scheduled(self) -> List[ScheduleAssignment]:
        return [a for a in self.assignments if not a.is_deferred]

    def deferred(self) -> List[ScheduleAssignment]:
        return [a for a in self.assignments if a.is_deferred]


# ─── Evaluation types ────────────────────────────────────────────────────────

@dataclass
class KPIResult:
    """Operational KPIs produced by evaluating a schedule against realized
    durations."""
    total_cost: float = 0.0
    activation_cost: float = 0.0
    overtime_cost: float = 0.0
    idle_cost: float = 0.0
    deferral_cost: float = 0.0
    overtime_minutes: float = 0.0
    idle_minutes: float = 0.0
    scheduled_count: int = 0
    deferred_count: int = 0
    blocks_opened: int = 0

    def summary_line(self) -> str:
        return (
            f"cost={self.total_cost:,.0f}  "
            f"act={self.activation_cost:,.0f}  "
            f"OT={self.overtime_minutes:.0f}m  "
            f"idle={self.idle_minutes:.0f}m  "
            f"sched={self.scheduled_count}  "
            f"def={self.deferred_count}  "
            f"blks={self.blocks_opened}"
        )
