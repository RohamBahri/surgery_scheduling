"""Canonical data types shared across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class Col:
    PATIENT_ID = "patient_id"
    PATIENT_TYPE = "patient_type"
    CASE_SERVICE = "case_service"
    MAIN_PROCEDURE = "main_procedure"
    PROCEDURE_ID = "main_procedure_id"
    OPERATING_ROOM = "operating_room"
    SURGEON = "surgeon"
    SURGEON_CODE = "surgeon_code"
    BOOKED_MINUTES = "booked_time_minutes"
    ACTUAL_START = "actual_start"
    ACTUAL_STOP = "actual_stop"
    ENTER_ROOM = "enter_room"
    LEAVE_ROOM = "leave_room"
    PROCEDURE_DURATION = "procedure_duration_min"
    PREPARATION_DURATION = "preparation_duration_min"
    WEEK_OF_YEAR = "week_of_year"
    MONTH = "month"
    YEAR = "year"

    SITE = "site"
    SURGICAL_DURATION = "surgical_duration_min"
    ROOM_DURATION = "room_duration_min"
    USED_ROOM_TIME = "used_room_time"
    FELL_BACK_SURGICAL = "fell_back_to_surgical"
    TIMESTAMP_VIOLATION = "timestamp_violation"
    OVERHEAD_CAPPED = "overhead_capped"
    CASE_UID = "case_uid"

    ACTUAL_START_DATE = "actual_start_date"
    ACTUAL_START_TIME = "actual_start_time"
    ACTUAL_STOP_DATE = "actual_stop_date"
    ACTUAL_STOP_TIME = "actual_stop_time"
    ENTER_ROOM_DATE = "enter_room_date"
    ENTER_ROOM_TIME = "enter_room_time"
    LEAVE_ROOM_DATE = "leave_room_date"
    LEAVE_ROOM_TIME = "leave_room_time"


class Domain:
    MIN_PROCEDURE_DURATION = 30
    MAX_OVERTIME_PER_BLOCK = 240
    OR_ROOM_PREFIX = "OR"
    EMERGENCY_PATIENT = "EMERGENCY PATIENT"
    OTHER = "Other"
    UNKNOWN = "Unknown"
    EMERGENCY_ROOMS = ("OREMER", "ORER")


class BlockId(NamedTuple):
    day_index: int
    site: str
    room: str


EligibilityMap = Dict[str, Set[Tuple[str, str]]]
SurgeonDaySiteCases = Dict[Tuple[str, int, str], List[int]]


@dataclass
class CaseRecord:
    case_id: int
    procedure_id: str
    surgeon_code: str
    service: str
    patient_type: str
    operating_room: str
    booked_duration_min: float
    actual_duration_min: float
    actual_start: datetime
    week_of_year: int
    month: int
    year: int
    site: str = ""
    surgical_duration_min: float = 0.0


@dataclass(frozen=True)
class CandidateBlock:
    day_index: int
    site: str
    room: str
    capacity_minutes: float
    activation_cost: float
    is_fixed: bool = False

    @property
    def id(self) -> BlockId:
        return BlockId(self.day_index, self.site, self.room)


@dataclass
class BlockCalendar:
    candidates: List[CandidateBlock]

    @property
    def block_ids(self) -> List[BlockId]:
        return [c.id for c in self.candidates]

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    @property
    def fixed_blocks(self) -> List[CandidateBlock]:
        return [c for c in self.candidates if c.is_fixed]

    @property
    def flex_blocks(self) -> List[CandidateBlock]:
        return [c for c in self.candidates if not c.is_fixed]

    def capacity(self, block_id: BlockId) -> float:
        for c in self.candidates:
            if c.id == block_id:
                return c.capacity_minutes
        raise KeyError(f"Block {block_id} not in calendar.")

    def activation_cost(self, block_id: BlockId) -> float:
        for c in self.candidates:
            if c.id == block_id:
                return c.activation_cost
        raise KeyError(f"Block {block_id} not in calendar.")

    def blocks_on_day(self, day: int) -> List[CandidateBlock]:
        return [c for c in self.candidates if c.day_index == day]

    def blocks_by_site_day(self, site: str, day: int) -> List[CandidateBlock]:
        return [c for c in self.candidates if c.site == site and c.day_index == day]


@dataclass
class WeeklyInstance:
    week_index: int
    start_date: date
    end_date: date
    cases: List[CaseRecord]
    calendar: BlockCalendar
    eligibility: EligibilityMap = field(default_factory=dict)
    case_eligible_blocks: Dict[int, List[BlockId]] = field(default_factory=dict)
    surgeon_day_site_cases: SurgeonDaySiteCases = field(default_factory=dict)

    @property
    def num_cases(self) -> int:
        return len(self.cases)

    def booked_durations(self) -> List[float]:
        return [c.booked_duration_min for c in self.cases]

    def actual_durations(self) -> List[float]:
        return [c.actual_duration_min for c in self.cases]


@dataclass
class ScheduleAssignment:
    case_id: int
    day_index: Optional[int] = None
    site: Optional[str] = None
    room: Optional[str] = None

    @property
    def is_deferred(self) -> bool:
        return self.day_index is None

    @property
    def block_id(self) -> Optional[BlockId]:
        if self.is_deferred:
            return None
        return BlockId(self.day_index, self.site or "", self.room or "")


@dataclass
class ScheduleResult:
    assignments: List[ScheduleAssignment]
    opened_blocks: Set[BlockId] = field(default_factory=set)
    solver_status: str = "Unknown"
    objective_value: Optional[float] = None
    solve_time_seconds: float = 0.0

    def scheduled(self) -> List[ScheduleAssignment]:
        return [a for a in self.assignments if not a.is_deferred]

    def deferred(self) -> List[ScheduleAssignment]:
        return [a for a in self.assignments if a.is_deferred]


@dataclass
class KPIResult:
    total_cost: float = 0.0
    activation_cost: float = 0.0
    overtime_cost: float = 0.0
    idle_cost: float = 0.0
    deferral_cost: float = 0.0
    overtime_minutes: float = 0.0
    idle_minutes: float = 0.0
    turnover_minutes: float = 0.0
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
