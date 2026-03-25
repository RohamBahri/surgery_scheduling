from datetime import date, datetime

from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, ScheduleAssignment, ScheduleResult, WeeklyInstance
from src.planning.audit import audit_surgeon_feasibility


def test_audit_surgeon_feasibility_smoke():
    cases = [
        CaseRecord(
            case_id=1,
            procedure_id="P1",
            surgeon_code="S1",
            service="SvcA",
            patient_type="ELECTIVE",
            operating_room="OR1",
            booked_duration_min=120,
            actual_duration_min=110,
            actual_start=datetime(2025, 1, 6, 8, 0, 0),
            week_of_year=2,
            month=1,
            year=2025,
            site="TGH",
            surgical_duration_min=100,
        )
    ]
    calendar = BlockCalendar(
        candidates=[CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=480, activation_cost=1000)]
    )
    instance = WeeklyInstance(
        week_index=0,
        start_date=date(2025, 1, 6),
        end_date=date(2025, 1, 12),
        cases=cases,
        calendar=calendar,
    )
    schedule = ScheduleResult(assignments=[ScheduleAssignment(case_id=1, day_index=0, site="TGH", room="OR1")])

    audit = audit_surgeon_feasibility(instance, schedule)
    assert audit.total_surgeon_day_sites >= 1
