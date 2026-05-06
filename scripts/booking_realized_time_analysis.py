"""Compare booked OR time with realized surgical, room, and support time.

This script intentionally has no third-party dependencies.  It reads the UHN
Excel workbook directly, keeps only non-cancelled cases with every required
timestamp present, and reports whether booked time over- or under-estimated the
realized duration by more than 30 minutes.

It also reports preparation, cleaning, and turnover-time diagnostics.  These
diagnostics are meant to test whether simple fixed or iid assumptions look
reasonable overall, or whether specialty-level distributions are more credible.

Usage:
    python3 scripts/booking_realized_time_analysis.py
    python3 scripts/booking_realized_time_analysis.py --data data/UHNOperating_RoomScheduling2011-2013.xlsx
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from html import escape
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile


DEFAULT_DATA = Path("data/UHNOperating_RoomScheduling2011-2013.xlsx")
DEFAULT_OUTPUT_DIR = Path("artifacts/booking_realized_time_analysis")

BOOKED_COL = "Booked Time (Minutes)"
PATIENT_TYPE_COL = "Patient_Type"
SPECIALTY_COL = "Case_Service"
ROOM_COL = "Operating_Room"
TIMESTAMP_COLS = [
    "Enter Room Date",
    "Enter Room Time",
    "Actual Start Date",
    "Actual Start Time",
    "Actual Stop Date",
    "Actual Stop Time",
    "Leave Room Date",
    "Leave Room Time",
]
CANCEL_COLS = ["Case_Cancelled_Reason", "Case Cancel Date", "Case Cancel Time"]
TIME_LAG_CLOSE_TO_FIXED_MINUTES = 15.0
MIN_SPECIALTY_CASES = 30
DEFAULT_HISTOGRAM_CLIP_MINUTES = 180.0
DEFAULT_HISTOGRAM_BIN_WIDTH = 10.0
DEFAULT_MAX_PLANNING_CASE_MINUTES = 480.0

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass
class AnalysisRow:
    row_number: int
    patient_id: str
    site: str
    operating_room: str
    patient_type: str
    case_service: str
    surgeon_code: str
    main_procedure_id: str
    main_procedure: str
    booked_minutes: float
    enter_room: datetime
    actual_start: datetime
    actual_stop: datetime
    leave_room: datetime
    room_minutes: float
    surgical_minutes: float

    @property
    def room_difference(self) -> float:
        return self.booked_minutes - self.room_minutes

    @property
    def surgical_difference(self) -> float:
        return self.booked_minutes - self.surgical_minutes

    @property
    def preparation_minutes(self) -> float:
        return (self.actual_start - self.enter_room).total_seconds() / 60.0

    @property
    def cleaning_minutes(self) -> float:
        return (self.leave_room - self.actual_stop).total_seconds() / 60.0


@dataclass
class TurnoverRow:
    room: str
    date: str
    previous_excel_row: int
    next_excel_row: int
    previous_specialty: str
    next_specialty: str
    previous_leave_room: datetime
    next_enter_room: datetime
    turnover_minutes: float

    @property
    def specialty(self) -> str:
        if self.previous_specialty == self.next_specialty:
            return self.previous_specialty
        return "MIXED_SPECIALTY"


@dataclass
class FilterStats:
    total_rows: int = 0
    or_room_rows: int = 0
    non_cancelled_rows: int = 0
    non_emergency_rows: int = 0
    booked_rows: int = 0
    complete_timestamp_rows: int = 0
    positive_duration_rows: int = 0
    planning_duration_rows: int = 0
    over_max_case_rows: int = 0
    ordered_rows: int = 0
    order_violation_rows: int = 0


def load_filtered_analysis_rows(
    data_path: Path = DEFAULT_DATA,
    sheet: str | None = None,
    *,
    allow_order_violations: bool = False,
    max_case_minutes: float = DEFAULT_MAX_PLANNING_CASE_MINUTES,
) -> tuple[list[AnalysisRow], FilterStats]:
    """Load cases using the common analysis and paper-figure filters."""
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    if max_case_minutes <= 0:
        raise SystemExit("max_case_minutes must be positive.")

    header, rows, date1904 = read_xlsx(data_path, sheet)
    validate_columns(header)

    stats = FilterStats()
    analysis_rows: list[AnalysisRow] = []

    for row_number, row in rows:
        stats.total_rows += 1

        if not is_exact_or_room(row):
            continue
        stats.or_room_rows += 1

        if not is_non_cancelled(row):
            continue
        stats.non_cancelled_rows += 1

        if is_emergency_case(row):
            continue
        stats.non_emergency_rows += 1

        booked = parse_number(row.get(BOOKED_COL))
        if booked is None or booked <= 0:
            continue
        stats.booked_rows += 1

        enter_room = combine_date_time(row, "Enter Room Date", "Enter Room Time", date1904)
        actual_start = combine_date_time(row, "Actual Start Date", "Actual Start Time", date1904)
        actual_stop = combine_date_time(row, "Actual Stop Date", "Actual Stop Time", date1904)
        leave_room = combine_date_time(row, "Leave Room Date", "Leave Room Time", date1904)
        if any(x is None for x in [enter_room, actual_start, actual_stop, leave_room]):
            continue
        stats.complete_timestamp_rows += 1

        room_minutes = minutes_between(enter_room, leave_room)
        surgical_minutes = minutes_between(actual_start, actual_stop)
        if room_minutes is None or surgical_minutes is None or room_minutes <= 0 or surgical_minutes <= 0:
            continue
        stats.positive_duration_rows += 1

        if booked > max_case_minutes or room_minutes > max_case_minutes or surgical_minutes > max_case_minutes:
            stats.over_max_case_rows += 1
            continue
        stats.planning_duration_rows += 1

        order_ok = enter_room <= actual_start <= actual_stop <= leave_room
        if not order_ok:
            stats.order_violation_rows += 1
            if not allow_order_violations:
                continue
        stats.ordered_rows += 1

        analysis_rows.append(
            AnalysisRow(
                row_number=row_number,
                patient_id=clean_text(row.get("Patient_ID")),
                site=clean_text(row.get("Site")),
                operating_room=clean_text(row.get("Operating_Room")),
                patient_type=clean_text(row.get("Patient_Type")),
                case_service=clean_text(row.get("Case_Service")),
                surgeon_code=clean_text(row.get("Surgeon_Code")),
                main_procedure_id=clean_text(row.get("Main_Procedure_Id")),
                main_procedure=clean_text(row.get("Main_Procedure")),
                booked_minutes=booked,
                enter_room=enter_room,
                actual_start=actual_start,
                actual_stop=actual_stop,
                leave_room=leave_room,
                room_minutes=room_minutes,
                surgical_minutes=surgical_minutes,
            )
        )

    return analysis_rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare booked OR time with realized room and surgical durations."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the .xlsx data file.")
    parser.add_argument("--sheet", default=None, help="Workbook sheet name. Defaults to the first sheet.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Minute threshold for over/under-estimation categories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--allow-order-violations",
        action="store_true",
        help=(
            "Keep rows with all timestamps and positive durations even if the full order "
            "Enter <= Start <= Stop <= Leave is violated."
        ),
    )
    parser.add_argument(
        "--histogram-clip",
        type=float,
        default=DEFAULT_HISTOGRAM_CLIP_MINUTES,
        help="Absolute x-axis clipping limit in minutes for the booked-room-time histogram.",
    )
    parser.add_argument(
        "--histogram-bin-width",
        type=float,
        default=DEFAULT_HISTOGRAM_BIN_WIDTH,
        help="Bin width in minutes for the booked-room-time histogram.",
    )
    parser.add_argument(
        "--max-case-minutes",
        type=float,
        default=DEFAULT_MAX_PLANNING_CASE_MINUTES,
        help=(
            "Drop cases whose booked time, room realized time, or surgical realized time "
            "exceeds this many minutes."
        ),
    )
    args = parser.parse_args()

    if args.histogram_clip <= 0:
        raise SystemExit("--histogram-clip must be positive.")
    if args.histogram_bin_width <= 0:
        raise SystemExit("--histogram-bin-width must be positive.")
    analysis_rows, filter_stats = load_filtered_analysis_rows(
        args.data,
        args.sheet,
        allow_order_violations=args.allow_order_violations,
        max_case_minutes=args.max_case_minutes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_path = args.output_dir / "case_level_differences.csv"
    summary_path = args.output_dir / "summary.csv"
    time_lag_path = args.output_dir / "time_lag_summary.csv"
    specialty_lag_path = args.output_dir / "time_lag_by_specialty.csv"
    turnover_path = args.output_dir / "turnover_case_level.csv"
    histogram_path = args.output_dir / "booked_minus_room_time_histogram.svg"

    write_case_csv(case_path, analysis_rows, args.threshold)
    summary_rows = [
        summarize_definition("room_time_enter_to_leave", [r.room_difference for r in analysis_rows], args.threshold),
        summarize_definition(
            "surgical_time_start_to_stop",
            [r.surgical_difference for r in analysis_rows],
            args.threshold,
        ),
    ]
    write_summary_csv(summary_path, summary_rows)
    turnover_rows = build_turnover_rows(analysis_rows)
    time_lag_rows = build_time_lag_summary_rows(analysis_rows, turnover_rows)
    specialty_lag_rows = build_specialty_lag_rows(analysis_rows, turnover_rows)
    write_summary_csv(time_lag_path, time_lag_rows)
    write_summary_csv(specialty_lag_path, specialty_lag_rows)
    write_turnover_csv(turnover_path, turnover_rows)
    write_room_difference_histogram_svg(
        histogram_path,
        [row.room_difference for row in analysis_rows],
        threshold=args.threshold,
        clip_minutes=args.histogram_clip,
        bin_width=args.histogram_bin_width,
    )

    print_report(
        data_path=args.data,
        sheet_name=args.sheet or "first sheet",
        threshold=args.threshold,
        total_rows=filter_stats.total_rows,
        or_room_rows=filter_stats.or_room_rows,
        non_cancelled_rows=filter_stats.non_cancelled_rows,
        non_emergency_rows=filter_stats.non_emergency_rows,
        booked_rows=filter_stats.booked_rows,
        complete_timestamp_rows=filter_stats.complete_timestamp_rows,
        positive_duration_rows=filter_stats.positive_duration_rows,
        planning_duration_rows=filter_stats.planning_duration_rows,
        over_max_case_rows=filter_stats.over_max_case_rows,
        max_case_minutes=args.max_case_minutes,
        ordered_rows=filter_stats.ordered_rows,
        order_violation_rows=filter_stats.order_violation_rows,
        allow_order_violations=args.allow_order_violations,
        analysis_rows=analysis_rows,
        summary_rows=summary_rows,
        time_lag_rows=time_lag_rows,
        specialty_lag_rows=specialty_lag_rows,
        turnover_rows=turnover_rows,
        case_path=case_path,
        summary_path=summary_path,
        time_lag_path=time_lag_path,
        specialty_lag_path=specialty_lag_path,
        turnover_path=turnover_path,
        histogram_path=histogram_path,
    )
    return 0


def read_xlsx(path: Path, sheet_name: str | None) -> tuple[list[str], list[tuple[int, dict[str, str]]], bool]:
    with ZipFile(path) as workbook:
        shared_strings = load_shared_strings(workbook)
        sheets = workbook_sheets(workbook)
        if not sheets:
            raise SystemExit(f"No worksheets found in {path}")

        if sheet_name is None:
            selected_name, selected_target = sheets[0]
        else:
            matches = [item for item in sheets if item[0] == sheet_name]
            if not matches:
                available = ", ".join(name for name, _ in sheets)
                raise SystemExit(f"Sheet {sheet_name!r} not found. Available sheets: {available}")
            selected_name, selected_target = matches[0]

        sheet_path = selected_target
        if not sheet_path.startswith("xl/"):
            sheet_path = "xl/" + sheet_path.lstrip("/")

        date1904 = workbook_uses_1904_dates(workbook)
        rows = parse_sheet_rows(workbook, sheet_path, shared_strings)
        if not rows:
            raise SystemExit(f"Sheet {selected_name!r} is empty.")

        header_values = rows[0][1]
        header = [clean_text(value) for value in header_values]
        data_rows: list[tuple[int, dict[str, str]]] = []
        for row_number, values in rows[1:]:
            row_dict: dict[str, str] = {}
            for idx, col_name in enumerate(header):
                if col_name:
                    row_dict[col_name] = values[idx] if idx < len(values) else ""
            data_rows.append((row_number, row_dict))

        return header, data_rows, date1904


def load_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(f"{NS_MAIN}si"):
        text = "".join(node.text or "" for node in item.findall(f".//{NS_MAIN}t"))
        strings.append(text)
    return strings


def workbook_sheets(workbook: ZipFile) -> list[tuple[str, str]]:
    rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

    wb_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    sheets = []
    for sheet in wb_root.findall(f".//{NS_MAIN}sheet"):
        rel_id = sheet.attrib[f"{NS_REL}id"]
        sheets.append((sheet.attrib["name"], rel_target[rel_id]))
    return sheets


def workbook_uses_1904_dates(workbook: ZipFile) -> bool:
    wb_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    workbook_pr = wb_root.find(f"{NS_MAIN}workbookPr")
    return workbook_pr is not None and workbook_pr.attrib.get("date1904") in {"1", "true", "True"}


def parse_sheet_rows(
    workbook: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[tuple[int, list[str]]]:
    root = ET.fromstring(workbook.read(sheet_path))
    parsed_rows = []
    for row in root.findall(f".//{NS_MAIN}sheetData/{NS_MAIN}row"):
        values: list[str] = []
        current_idx = 0
        for cell in row.findall(f"{NS_MAIN}c"):
            idx = column_index(cell.attrib.get("r", "A1"))
            while current_idx < idx:
                values.append("")
                current_idx += 1
            values.append(cell_value(cell, shared_strings))
            current_idx += 1
        parsed_rows.append((int(row.attrib.get("r", len(parsed_rows) + 1)), values))
    return parsed_rows


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for letter in letters:
        idx = idx * 26 + ord(letter.upper()) - ord("A") + 1
    return idx - 1


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find(f"{NS_MAIN}v")
    if value_node is None:
        inline = cell.find(f"{NS_MAIN}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.findall(f".//{NS_MAIN}t"))

    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    return raw


def validate_columns(header: list[str]) -> None:
    required = [BOOKED_COL, PATIENT_TYPE_COL, SPECIALTY_COL, ROOM_COL, *TIMESTAMP_COLS, *CANCEL_COLS]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))


def is_non_cancelled(row: dict[str, str]) -> bool:
    return all(is_blank(row.get(col)) for col in CANCEL_COLS)


def is_exact_or_room(row: dict[str, str]) -> bool:
    room = clean_text(row.get(ROOM_COL)).upper()
    compact = "".join(room.split())
    return compact.startswith("OR") and compact[2:].isdigit()


def is_emergency_case(row: dict[str, str]) -> bool:
    patient_type = clean_text(row.get(PATIENT_TYPE_COL)).lower()
    compact = " ".join(patient_type.replace("_", " ").replace("-", " ").split())
    return compact in {"emergency", "emergency patient"}


def is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "nat"}


def clean_text(value: object) -> str:
    return "" if is_blank(value) else str(value).strip()


def parse_number(value: object) -> float | None:
    if is_blank(value):
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def combine_date_time(
    row: dict[str, str],
    date_col: str,
    time_col: str,
    date1904: bool,
) -> datetime | None:
    date_value = parse_excel_date(row.get(date_col), date1904)
    time_value = parse_excel_time(row.get(time_col))
    if date_value is None or time_value is None:
        return None
    return datetime.combine(date_value.date(), time_value)


def parse_excel_date(value: object, date1904: bool) -> datetime | None:
    if is_blank(value):
        return None

    number = parse_number(value)
    if number is not None:
        origin = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
        return origin + timedelta(days=number)

    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_excel_time(value: object) -> time | None:
    if is_blank(value):
        return None

    number = parse_number(value)
    if number is not None:
        fraction = number % 1
        seconds = int(round(fraction * 24 * 60 * 60))
        seconds %= 24 * 60 * 60
        return (datetime.min + timedelta(seconds=seconds)).time()

    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass

    try:
        parsed = datetime.fromisoformat(text)
        return parsed.time()
    except ValueError:
        return None


def minutes_between(start: datetime, stop: datetime) -> float | None:
    minutes = (stop - start).total_seconds() / 60.0
    return minutes if math.isfinite(minutes) else None


def is_same_day_case(row: AnalysisRow) -> bool:
    """True when all intra-case timestamp lags occur within one calendar day."""
    day = row.enter_room.date()
    return (
        row.actual_start.date() == day
        and row.actual_stop.date() == day
        and row.leave_room.date() == day
    )


def same_day_case_rows(rows: list[AnalysisRow]) -> list[AnalysisRow]:
    return [row for row in rows if is_same_day_case(row)]


def summarize_definition(name: str, differences: list[float], threshold: float) -> dict[str, float | int | str]:
    if not differences:
        return {
            "definition": name,
            "n": 0,
            "mean_difference_booked_minus_realized": "",
            "median_difference_booked_minus_realized": "",
            "mean_absolute_error": "",
            "overestimated_gt_threshold_n": 0,
            "overestimated_gt_threshold_pct": "",
            "underestimated_gt_threshold_n": 0,
            "underestimated_gt_threshold_pct": "",
            "within_threshold_n": 0,
            "within_threshold_pct": "",
            "p10_difference": "",
            "p25_difference": "",
            "p75_difference": "",
            "p90_difference": "",
        }

    n = len(differences)
    over = sum(diff > threshold for diff in differences)
    under = sum(diff < -threshold for diff in differences)
    within = n - over - under
    return {
        "definition": name,
        "n": n,
        "mean_difference_booked_minus_realized": statistics.fmean(differences),
        "median_difference_booked_minus_realized": statistics.median(differences),
        "mean_absolute_error": statistics.fmean(abs(diff) for diff in differences),
        "overestimated_gt_threshold_n": over,
        "overestimated_gt_threshold_pct": 100 * over / n,
        "underestimated_gt_threshold_n": under,
        "underestimated_gt_threshold_pct": 100 * under / n,
        "within_threshold_n": within,
        "within_threshold_pct": 100 * within / n,
        "p10_difference": percentile(differences, 10),
        "p25_difference": percentile(differences, 25),
        "p75_difference": percentile(differences, 75),
        "p90_difference": percentile(differences, 90),
    }


def build_turnover_rows(rows: list[AnalysisRow]) -> list[TurnoverRow]:
    by_room_day: dict[tuple[str, str], list[AnalysisRow]] = {}
    for row in rows:
        if not is_same_day_case(row):
            continue
        room = row.operating_room or "UNKNOWN_ROOM"
        day = row.enter_room.date().isoformat()
        by_room_day.setdefault((room, day), []).append(row)

    turnovers: list[TurnoverRow] = []
    for (room, day), day_rows in by_room_day.items():
        ordered = sorted(day_rows, key=lambda item: (item.enter_room, item.leave_room, item.row_number))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.leave_room.date() != current.enter_room.date():
                continue
            turnover = minutes_between(previous.leave_room, current.enter_room)
            if turnover is None or turnover < 0:
                continue
            turnovers.append(
                TurnoverRow(
                    room=room,
                    date=day,
                    previous_excel_row=previous.row_number,
                    next_excel_row=current.row_number,
                    previous_specialty=previous.case_service or "UNKNOWN",
                    next_specialty=current.case_service or "UNKNOWN",
                    previous_leave_room=previous.leave_room,
                    next_enter_room=current.enter_room,
                    turnover_minutes=turnover,
                )
            )
    return turnovers


def build_time_lag_summary_rows(
    case_rows: list[AnalysisRow],
    turnover_rows: list[TurnoverRow],
) -> list[dict[str, float | int | str]]:
    same_day_rows = same_day_case_rows(case_rows)
    lag_specs = [
        (
            "preparation_enter_to_start",
            [(row.preparation_minutes, row.case_service or "UNKNOWN") for row in same_day_rows],
        ),
        (
            "cleaning_stop_to_leave",
            [(row.cleaning_minutes, row.case_service or "UNKNOWN") for row in same_day_rows],
        ),
        (
            "turnover_leave_to_next_enter",
            [(row.turnover_minutes, row.specialty or "UNKNOWN") for row in turnover_rows],
        ),
    ]
    serial_corr = serial_lag_correlations(same_day_rows, turnover_rows)
    output = []
    for lag_type, values in lag_specs:
        row = summarize_lag(lag_type, values)
        row["same_room_day_serial_corr"] = serial_corr.get(lag_type, "")
        row["iid_read"] = iid_read(row)
        output.append(row)
    return output


def build_specialty_lag_rows(
    case_rows: list[AnalysisRow],
    turnover_rows: list[TurnoverRow],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in same_day_case_rows(case_rows):
        grouped.setdefault(("preparation_enter_to_start", row.case_service or "UNKNOWN"), []).append(
            row.preparation_minutes
        )
        grouped.setdefault(("cleaning_stop_to_leave", row.case_service or "UNKNOWN"), []).append(
            row.cleaning_minutes
        )
    for row in turnover_rows:
        grouped.setdefault(("turnover_leave_to_next_enter", row.specialty or "UNKNOWN"), []).append(
            row.turnover_minutes
        )

    output = []
    for (lag_type, specialty), values in grouped.items():
        clean_values = valid_lag_values(values)
        if len(clean_values) < MIN_SPECIALTY_CASES:
            continue
        row = basic_lag_stats(clean_values)
        row["lag_type"] = lag_type
        row["specialty"] = specialty
        output.append(row)

    output.sort(key=lambda row: (str(row["lag_type"]), -int(row["n"]), str(row["specialty"])))
    return output


def summarize_lag(lag_type: str, values_with_specialty: list[tuple[float, str]]) -> dict[str, float | int | str]:
    values = valid_lag_values(value for value, _ in values_with_specialty)
    row = basic_lag_stats(values)
    row["lag_type"] = lag_type
    row["specialty_eta_squared"] = specialty_eta_squared(values_with_specialty)
    row["fixed_time_read"] = fixed_time_read(row)
    row["specialty_read"] = specialty_read(row["specialty_eta_squared"])
    return row


def valid_lag_values(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value >= 0]


def basic_lag_stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "n": 0,
            "mean": "",
            "median": "",
            "std": "",
            "cv": "",
            "p10": "",
            "p25": "",
            "p75": "",
            "p90": "",
            "p95": "",
            "p99": "",
            "max": "",
            "iqr": "",
            "pct_within_15_min_of_median": "",
        }

    median = statistics.median(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    mean = statistics.fmean(values)
    p25 = percentile(values, 25)
    p75 = percentile(values, 75)
    close = sum(abs(value - median) <= TIME_LAG_CLOSE_TO_FIXED_MINUTES for value in values)
    return {
        "n": len(values),
        "mean": mean,
        "median": median,
        "std": std,
        "cv": std / mean if mean > 0 else "",
        "p10": percentile(values, 10),
        "p25": p25,
        "p75": p75,
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "iqr": p75 - p25,
        "pct_within_15_min_of_median": 100 * close / len(values),
    }


def serial_lag_correlations(
    case_rows: list[AnalysisRow],
    turnover_rows: list[TurnoverRow],
) -> dict[str, float | str]:
    by_room_day: dict[tuple[str, str], list[AnalysisRow]] = {}
    for row in case_rows:
        by_room_day.setdefault((row.operating_room, row.enter_room.date().isoformat()), []).append(row)

    prep_pairs: list[tuple[float, float]] = []
    cleaning_pairs: list[tuple[float, float]] = []
    for rows in by_room_day.values():
        ordered = sorted(rows, key=lambda item: (item.enter_room, item.row_number))
        prep_values = [row.preparation_minutes for row in ordered]
        cleaning_values = [row.cleaning_minutes for row in ordered]
        prep_pairs.extend(zip(prep_values, prep_values[1:]))
        cleaning_pairs.extend(zip(cleaning_values, cleaning_values[1:]))

    by_turnover_room_day: dict[tuple[str, str], list[TurnoverRow]] = {}
    for row in turnover_rows:
        by_turnover_room_day.setdefault((row.room, row.date), []).append(row)

    turnover_pairs: list[tuple[float, float]] = []
    for rows in by_turnover_room_day.values():
        ordered = sorted(rows, key=lambda item: (item.previous_leave_room, item.next_enter_room))
        values = [row.turnover_minutes for row in ordered]
        turnover_pairs.extend(zip(values, values[1:]))

    return {
        "preparation_enter_to_start": pearson_pair_corr(prep_pairs),
        "cleaning_stop_to_leave": pearson_pair_corr(cleaning_pairs),
        "turnover_leave_to_next_enter": pearson_pair_corr(turnover_pairs),
    }


def pearson_pair_corr(pairs: list[tuple[float, float]]) -> float | str:
    pairs = [
        (x, y)
        for x, y in pairs
        if math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0
    ]
    if len(pairs) < 3:
        return ""
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return ""
    return cov / math.sqrt(var_x * var_y)


def specialty_eta_squared(values_with_specialty: list[tuple[float, str]]) -> float | str:
    grouped: dict[str, list[float]] = {}
    for value, specialty in values_with_specialty:
        if math.isfinite(value) and value >= 0:
            grouped.setdefault(specialty or "UNKNOWN", []).append(value)

    grouped = {key: values for key, values in grouped.items() if len(values) >= MIN_SPECIALTY_CASES}
    values = [value for group_values in grouped.values() for value in group_values]
    if len(values) < 2 or len(grouped) < 2:
        return ""

    grand_mean = statistics.fmean(values)
    total_ss = sum((value - grand_mean) ** 2 for value in values)
    if total_ss <= 0:
        return 0.0

    between_ss = 0.0
    for group_values in grouped.values():
        group_mean = statistics.fmean(group_values)
        between_ss += len(group_values) * ((group_mean - grand_mean) ** 2)
    return between_ss / total_ss


def fixed_time_read(row: dict[str, float | int | str]) -> str:
    if int(row["n"]) == 0:
        return "no data"
    pct_close = float(row["pct_within_15_min_of_median"])
    iqr = float(row["iqr"])
    cv = row["cv"]
    cv_value = float(cv) if cv != "" else float("inf")
    if pct_close >= 80 and iqr <= 20 and cv_value <= 0.5:
        return "fixed-time approximation looks plausible"
    if pct_close >= 80 and iqr <= 20:
        return "typical fixed time looks plausible, but outliers matter"
    if pct_close >= 60 and iqr <= 35:
        return "fixed-time approximation is rough"
    return "fixed-time approximation is weak"


def specialty_read(eta: float | str) -> str:
    if eta == "":
        return "not enough specialty groups"
    eta_value = float(eta)
    if eta_value >= 0.10:
        return "strong specialty dependence"
    if eta_value >= 0.03:
        return "moderate specialty dependence"
    if eta_value >= 0.01:
        return "small specialty dependence"
    return "little specialty dependence"


def iid_read(row: dict[str, float | int | str]) -> str:
    eta = row.get("specialty_eta_squared", "")
    serial_corr = row.get("same_room_day_serial_corr", "")
    eta_value = float(eta) if eta != "" else 0.0
    serial_value = abs(float(serial_corr)) if serial_corr != "" else 0.0
    if eta_value >= 0.10 or serial_value >= 0.20:
        return "pooled iid assumption is weak"
    if eta_value >= 0.03 or serial_value >= 0.10:
        return "pooled iid assumption is rough"
    return "pooled iid assumption looks reasonable as a first approximation"


def percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def category(diff: float, threshold: float) -> str:
    if diff > threshold:
        return f"overestimated_gt_{threshold:g}_min"
    if diff < -threshold:
        return f"underestimated_gt_{threshold:g}_min"
    return f"within_{threshold:g}_min"


def write_case_csv(path: Path, rows: list[AnalysisRow], threshold: float) -> None:
    fields = [
        "excel_row",
        "patient_id",
        "site",
        "operating_room",
        "patient_type",
        "case_service",
        "surgeon_code",
        "main_procedure_id",
        "main_procedure",
        "booked_minutes",
        "enter_room",
        "actual_start",
        "actual_stop",
        "leave_room",
        "room_minutes_enter_to_leave",
        "surgical_minutes_start_to_stop",
        "same_day_for_lag_analysis",
        "preparation_minutes_enter_to_start",
        "cleaning_minutes_stop_to_leave",
        "room_difference_booked_minus_realized",
        "room_category",
        "surgical_difference_booked_minus_realized",
        "surgical_category",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            same_day_lag = is_same_day_case(row)
            writer.writerow(
                {
                    "excel_row": row.row_number,
                    "patient_id": row.patient_id,
                    "site": row.site,
                    "operating_room": row.operating_room,
                    "patient_type": row.patient_type,
                    "case_service": row.case_service,
                    "surgeon_code": row.surgeon_code,
                    "main_procedure_id": row.main_procedure_id,
                    "main_procedure": row.main_procedure,
                    "booked_minutes": round(row.booked_minutes, 3),
                    "enter_room": row.enter_room.isoformat(sep=" "),
                    "actual_start": row.actual_start.isoformat(sep=" "),
                    "actual_stop": row.actual_stop.isoformat(sep=" "),
                    "leave_room": row.leave_room.isoformat(sep=" "),
                    "room_minutes_enter_to_leave": round(row.room_minutes, 3),
                    "surgical_minutes_start_to_stop": round(row.surgical_minutes, 3),
                    "same_day_for_lag_analysis": same_day_lag,
                    "preparation_minutes_enter_to_start": round(row.preparation_minutes, 3) if same_day_lag else "",
                    "cleaning_minutes_stop_to_leave": round(row.cleaning_minutes, 3) if same_day_lag else "",
                    "room_difference_booked_minus_realized": round(row.room_difference, 3),
                    "room_category": category(row.room_difference, threshold),
                    "surgical_difference_booked_minus_realized": round(row.surgical_difference, 3),
                    "surgical_category": category(row.surgical_difference, threshold),
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 3) if isinstance(value, float) else value for key, value in row.items()})


def write_turnover_csv(path: Path, rows: list[TurnoverRow]) -> None:
    fields = [
        "room",
        "date",
        "previous_excel_row",
        "next_excel_row",
        "previous_specialty",
        "next_specialty",
        "turnover_specialty_group",
        "previous_leave_room",
        "next_enter_room",
        "turnover_minutes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "room": row.room,
                    "date": row.date,
                    "previous_excel_row": row.previous_excel_row,
                    "next_excel_row": row.next_excel_row,
                    "previous_specialty": row.previous_specialty,
                    "next_specialty": row.next_specialty,
                    "turnover_specialty_group": row.specialty,
                    "previous_leave_room": row.previous_leave_room.isoformat(sep=" "),
                    "next_enter_room": row.next_enter_room.isoformat(sep=" "),
                    "turnover_minutes": round(row.turnover_minutes, 3),
                }
            )


def write_room_difference_histogram_svg(
    path: Path,
    differences: list[float],
    *,
    threshold: float,
    clip_minutes: float,
    bin_width: float,
) -> None:
    values = [value for value in differences if math.isfinite(value)]
    width = 900
    height = 520
    margin_left = 76
    margin_right = 40
    margin_top = 30
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = -clip_minutes
    x_max = clip_minutes
    n_bins = max(1, int(math.ceil((x_max - x_min) / bin_width)))
    actual_bin_width = (x_max - x_min) / n_bins
    counts = [0 for _ in range(n_bins)]
    for value in values:
        clipped = min(max(value, x_min), x_max)
        idx = int((clipped - x_min) / actual_bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    max_count = max(counts) if counts else 0
    n = len(values)
    mae = statistics.fmean(abs(value) for value in values) if values else 0.0
    within = sum(abs(value) <= threshold for value in values)
    within_pct = 100 * within / n if n else 0.0

    def x_coord(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_coord(count: int) -> float:
        if max_count <= 0:
            return margin_top + plot_height
        return margin_top + plot_height - (count / max_count) * plot_height

    tick_step = 30 if clip_minutes <= 180 else 60
    ticks = []
    tick = math.ceil(x_min / tick_step) * tick_step
    while tick <= x_max + 1e-9:
        ticks.append(tick)
        tick += tick_step

    band_left = x_coord(max(-threshold, x_min))
    band_right = x_coord(min(threshold, x_max))
    zero_x = x_coord(0)
    annotation_x = margin_left + plot_width - 198
    annotation_y = margin_top + 18

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    svg.append(
        f'<rect x="{band_left:.2f}" y="{margin_top}" width="{band_right - band_left:.2f}" '
        f'height="{plot_height}" fill="#c8d8e8" opacity="0.45"/>'
    )

    for i, count in enumerate(counts):
        x0 = margin_left + i * (plot_width / n_bins)
        bar_width = max(0.5, plot_width / n_bins - 1)
        y0 = y_coord(count)
        bar_height = margin_top + plot_height - y0
        svg.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            'fill="#4f7693" opacity="0.88"/>'
        )

    svg.append(
        f'<line x1="{zero_x:.2f}" y1="{margin_top}" x2="{zero_x:.2f}" '
        f'y2="{margin_top + plot_height}" stroke="#111111" stroke-width="1.8"/>'
    )

    x_axis_y = margin_top + plot_height
    svg.append(
        f'<line x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left + plot_width}" '
        f'y2="{x_axis_y}" stroke="#1f2933" stroke-width="1"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{x_axis_y}" stroke="#1f2933" stroke-width="1"/>'
    )
    for tick in ticks:
        x = x_coord(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{x_axis_y}" x2="{x:.2f}" y2="{x_axis_y + 6}" stroke="#1f2933"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{x_axis_y + 23}" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#3e4c59" text-anchor="middle">{tick:g}</text>'
        )
    svg.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 32}" '
        'font-family="Arial, sans-serif" font-size="13" fill="#111111" text-anchor="middle">'
        'Booked - room time (minutes)</text>'
    )
    svg.append(
        f'<text x="22" y="{margin_top + plot_height / 2:.2f}" font-family="Arial, sans-serif" '
        'font-size="13" fill="#111111" text-anchor="middle" transform="rotate(-90 22 '
        f'{margin_top + plot_height / 2:.2f})">Number of cases</text>'
    )

    for frac in [0.25, 0.5, 0.75, 1.0]:
        count = int(round(max_count * frac))
        y = y_coord(count)
        svg.append(
            f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" '
            'stroke="#1f2933"/>'
        )
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
            f'font-size="10" fill="#3e4c59" text-anchor="end">{count:,}</text>'
        )

    svg.append(
        f'<rect x="{annotation_x}" y="{annotation_y}" width="182" height="78" '
        'fill="#ffffff" stroke="#c9d1d9" stroke-width="1"/>'
    )
    annotation_lines = [
        f"n = {n:,}",
        f"MAE = {mae:.1f} min",
        f"within +/-{threshold:g} min = {within_pct:.1f}%",
    ]
    for i, line in enumerate(annotation_lines):
        svg.append(
            f'<text x="{annotation_x + 12}" y="{annotation_y + 22 + i * 20}" '
            'font-family="Arial, sans-serif" font-size="13" fill="#111111">'
            f'{escape(line)}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def print_report(
    *,
    data_path: Path,
    sheet_name: str,
    threshold: float,
    total_rows: int,
    or_room_rows: int,
    non_cancelled_rows: int,
    non_emergency_rows: int,
    booked_rows: int,
    complete_timestamp_rows: int,
    positive_duration_rows: int,
    planning_duration_rows: int,
    over_max_case_rows: int,
    max_case_minutes: float,
    ordered_rows: int,
    order_violation_rows: int,
    allow_order_violations: bool,
    analysis_rows: list[AnalysisRow],
    summary_rows: list[dict[str, float | int | str]],
    time_lag_rows: list[dict[str, float | int | str]],
    specialty_lag_rows: list[dict[str, float | int | str]],
    turnover_rows: list[TurnoverRow],
    case_path: Path,
    summary_path: Path,
    time_lag_path: Path,
    specialty_lag_path: Path,
    turnover_path: Path,
    histogram_path: Path,
) -> None:
    print("\nBooked vs. Realized OR Time Analysis")
    print("=" * 42)
    print(f"Data file: {data_path}")
    print(f"Sheet: {sheet_name}")
    print(f"Difference convention: booked minutes - realized minutes")
    print(f"Threshold: {threshold:g} minutes")
    print()
    print("Sequential filters")
    print(f"  Raw rows:                         {total_rows:,}")
    print(f"  Exact OR\\d+ rooms:                {or_room_rows:,}")
    print(f"  Non-cancelled rows:               {non_cancelled_rows:,}")
    print(f"  Non-emergency rows:               {non_emergency_rows:,}")
    print(f"  With positive booked time:         {booked_rows:,}")
    print(f"  With all 4 timestamp pairs:        {complete_timestamp_rows:,}")
    print(f"  With positive realized durations:  {positive_duration_rows:,}")
    print(f"  Dropped > {max_case_minutes:g} min booked/realized:{over_max_case_rows:>8,}")
    print(f"  Within planning duration limit:    {planning_duration_rows:,}")
    if allow_order_violations:
        print(f"  Timestamp order violations kept:   {order_violation_rows:,}")
        print(f"  Final analysis rows:               {len(analysis_rows):,}")
    else:
        print(f"  Timestamp order violations dropped:{order_violation_rows:>9,}")
        print(f"  Final analysis rows:               {ordered_rows:,}")
    print()
    print("Results")
    for row in summary_rows:
        print_summary_row(row, threshold)

    print("\nPreparation, Cleaning, and Turnover Diagnostics")
    print("  Preparation/cleaning use only cases whose enter/start/stop/leave timestamps are on the same calendar day.")
    print("  Turnover uses same operating room, same calendar day, consecutive enter-room times.")
    print(f"  Turnover observations: {len(turnover_rows):,}")
    for row in time_lag_rows:
        print_lag_summary_row(row)

    print("\nSpecialty dependence check")
    print(f"  Specialty groups use {SPECIALTY_COL}; groups with < {MIN_SPECIALTY_CASES} observations are omitted.")
    for row in time_lag_rows:
        print(
            f"  {row['lag_type']}: eta^2={format_optional_float(row['specialty_eta_squared'])} "
            f"({row['specialty_read']})"
        )
    print_specialty_extremes(specialty_lag_rows)

    print(f"\nSaved case-level CSV:       {case_path}")
    print(f"Saved booked summary CSV:   {summary_path}")
    print(f"Saved lag summary CSV:      {time_lag_path}")
    print(f"Saved lag-by-specialty CSV: {specialty_lag_path}")
    print(f"Saved turnover CSV:         {turnover_path}")
    print(f"Saved histogram SVG:        {histogram_path}")


def print_summary_row(row: dict[str, float | int | str], threshold: float) -> None:
    name = str(row["definition"])
    n = int(row["n"])
    print(f"  {name}")
    if n == 0:
        print("    No analyzable cases.")
        return
    print(f"    n: {n:,}")
    print(f"    mean booked-realized:   {float(row['mean_difference_booked_minus_realized']):8.1f} min")
    print(f"    median booked-realized: {float(row['median_difference_booked_minus_realized']):8.1f} min")
    print(f"    mean absolute error:    {float(row['mean_absolute_error']):8.1f} min")
    print(
        f"    overestimated > {threshold:g} min:  "
        f"{int(row['overestimated_gt_threshold_n']):,} "
        f"({float(row['overestimated_gt_threshold_pct']):.1f}%)"
    )
    print(
        f"    underestimated > {threshold:g} min:"
        f" {int(row['underestimated_gt_threshold_n']):,} "
        f"({float(row['underestimated_gt_threshold_pct']):.1f}%)"
    )
    print(
        f"    within +/- {threshold:g} min:       "
        f"{int(row['within_threshold_n']):,} "
        f"({float(row['within_threshold_pct']):.1f}%)"
    )


def print_lag_summary_row(row: dict[str, float | int | str]) -> None:
    n = int(row["n"])
    print(f"  {row['lag_type']}")
    if n == 0:
        print("    No analyzable observations.")
        return
    print(f"    n: {n:,}")
    print(f"    mean / median:       {float(row['mean']):6.1f} / {float(row['median']):6.1f} min")
    print(f"    std / IQR:           {float(row['std']):6.1f} / {float(row['iqr']):6.1f} min")
    print(f"    p10 / p90:           {float(row['p10']):6.1f} / {float(row['p90']):6.1f} min")
    print(f"    p95 / p99 / max:     {float(row['p95']):6.1f} / {float(row['p99']):6.1f} / {float(row['max']):6.1f} min")
    print(
        f"    within +/- {TIME_LAG_CLOSE_TO_FIXED_MINUTES:g} min of median: "
        f"{float(row['pct_within_15_min_of_median']):.1f}%"
    )
    print(f"    same-room-day serial corr: {format_optional_float(row['same_room_day_serial_corr'])}")
    print(f"    fixed-time read:     {row['fixed_time_read']}")
    print(f"    iid read:            {row['iid_read']}")


def print_specialty_extremes(rows: list[dict[str, float | int | str]]) -> None:
    for lag_type in [
        "preparation_enter_to_start",
        "cleaning_stop_to_leave",
        "turnover_leave_to_next_enter",
    ]:
        selected = [row for row in rows if row["lag_type"] == lag_type]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: float(row["median"]))
        low = selected[0]
        high = selected[-1]
        print(
            f"  {lag_type}: specialty median range "
            f"{low['specialty']}={float(low['median']):.1f} min to "
            f"{high['specialty']}={float(high['median']):.1f} min"
        )


def format_optional_float(value: float | int | str) -> str:
    if value == "":
        return "N/A"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
