"""Generate paper-ready figures from the UHN surgery scheduling data.

Figure 1:
    Histogram of booking deviation, defined as booked time minus realized room
    time.  The caption is set in LaTeX, not on the figure.

Figure 2:
    Two-panel weekly realized overtime/idle-minute profile comparing the
    oracle and the status-quo plan.

All figures in this script must use the same case filtering rules as
scripts.booking_realized_time_analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.booking_realized_time_analysis import (
    DEFAULT_DATA,
    DEFAULT_MAX_PLANNING_CASE_MINUTES,
    load_filtered_analysis_rows,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/paper_figures")
DEFAULT_FIGURE1_PATH = DEFAULT_OUTPUT_DIR / "figure1_booking_deviation_hist.pdf"
DEFAULT_FIGURE2_PATH = DEFAULT_OUTPUT_DIR / "figure2_oracle_statusquo_minutes.pdf"

WEEK_COLUMN = "week"
PAIRED_WEEKLY_COLUMNS = (
    "realized_overtime_minutes__Oracle",
    "realized_overtime_minutes__StatusQuo",
    "realized_idle_minutes__Oracle",
    "realized_idle_minutes__StatusQuo",
)


# -----------------------------------------------------------------------------
# Journal-ready style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_deviations(
    data_path: Path = DEFAULT_DATA,
    sheet: str | None = None,
    *,
    max_case_minutes: float = DEFAULT_MAX_PLANNING_CASE_MINUTES,
) -> np.ndarray:
    """Return booked minus realized room time under the common filters."""
    rows, _ = load_filtered_analysis_rows(
        data_path,
        sheet,
        allow_order_violations=False,
        max_case_minutes=max_case_minutes,
    )
    return np.asarray([row.room_difference for row in rows], dtype=float)


def plot_booking_deviation(
    diffs: np.ndarray,
    output_path: Path | str,
    *,
    threshold: float = 30,
    x_clip: float = 480,
    bin_width: float = 5,
) -> None:
    """Histogram of booked minus realized room time."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        raise ValueError("No finite booking deviations available for plotting.")

    mae = np.mean(np.abs(diffs))
    within = np.mean(np.abs(diffs) <= threshold) * 100
    diffs_display = np.clip(diffs, -x_clip, x_clip)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)

    ax.axvspan(-threshold, threshold, color="0.85", alpha=0.5, zorder=0)
    bins = np.arange(-x_clip, x_clip + bin_width, bin_width)
    ax.hist(
        diffs_display,
        bins=bins,
        color="#4a6fa5",
        edgecolor="white",
        linewidth=0.3,
        zorder=1,
    )
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)

    ax.set_xlabel(r"Booked - realized room time (minutes)")
    ax.set_ylabel("Number of cases")
    ax.set_xlim(-x_clip, x_clip)
    ax.set_xticks(np.arange(-x_clip, x_clip + 1, 60))

    annotation = (
        f"n = {n:,}\n"
        f"MAE = {mae:.0f} min\n"
    )
    ax.text(
        0.97,
        0.97,
        annotation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7",
            linewidth=0.5,
        ),
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved {output_path}")


def _csv_has_columns(path: Path, columns: tuple[str, ...]) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return set(columns).issubset(header)


def find_latest_paired_weekly_path(
    search_root: Path = ROOT / "artifacts" / "experiments",
) -> Path | None:
    """Return the newest paired-weekly artifact with the columns needed here."""
    candidates = sorted(
        search_root.glob("*/paired_weekly_deltas.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if _csv_has_columns(path, PAIRED_WEEKLY_COLUMNS):
            return path
    return None


def load_paired_weekly_minutes(csv_path: Path) -> dict[str, np.ndarray]:
    """Load weekly Oracle and status-quo realized overtime/idle minutes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Paired weekly deltas not found: {csv_path}")

    values: dict[str, list[float]] = {col: [] for col in PAIRED_WEEKLY_COLUMNS}
    weeks: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [col for col in PAIRED_WEEKLY_COLUMNS if col not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

        for row_number, row in enumerate(reader, start=2):
            for col in PAIRED_WEEKLY_COLUMNS:
                raw_value = row.get(col, "")
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value in {csv_path} at row {row_number}, "
                        f"column {col}: {raw_value!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value in {csv_path} at row {row_number}, "
                        f"column {col}: {raw_value!r}"
                    )
                values[col].append(value)

            if WEEK_COLUMN in fieldnames:
                raw_week = row.get(WEEK_COLUMN, "")
                try:
                    week = float(raw_week)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value in {csv_path} at row {row_number}, "
                        f"column {WEEK_COLUMN}: {raw_week!r}"
                    ) from exc
                if not math.isfinite(week):
                    raise ValueError(
                        f"Non-finite value in {csv_path} at row {row_number}, "
                        f"column {WEEK_COLUMN}: {raw_week!r}"
                    )
            else:
                week = float(len(weeks) + 1)
            weeks.append(week)

    if not values[PAIRED_WEEKLY_COLUMNS[0]]:
        raise ValueError(f"No weekly rows found in {csv_path}")

    paired_minutes = {col: np.asarray(col_values, dtype=float) for col, col_values in values.items()}
    week_values = np.asarray(weeks, dtype=float)
    if np.array_equal(week_values, np.arange(len(week_values), dtype=float)):
        week_values = week_values + 1
    paired_minutes[WEEK_COLUMN] = week_values
    return paired_minutes


def _paired_axis_upper(x: np.ndarray, y: np.ndarray) -> float:
    data_min = min(float(np.min(x)), float(np.min(y)))
    data_max = max(float(np.max(x)), float(np.max(y)))
    if data_min < 0:
        raise ValueError("Physical minute values must be nonnegative.")
    if data_max == 0:
        return 1.0
    return 1.1 * data_max


def _plot_weekly_minutes_panel(
    ax: plt.Axes,
    weeks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> None:
    max_axis = _paired_axis_upper(x, y)
    week_min = float(np.min(weeks))
    week_max = float(np.max(weeks))
    week_ticks = [week_min]
    interior_ticks = [float(tick) for tick in np.arange(20, week_max, 20) if tick > week_min]
    if interior_ticks and week_max - interior_ticks[-1] < 10:
        interior_ticks = interior_ticks[:-1]
    week_ticks.extend(interior_ticks)
    if week_ticks[-1] != week_max:
        week_ticks.append(week_max)

    ax.vlines(
        weeks,
        x,
        y,
        color="0.72",
        linewidth=0.45,
        alpha=0.55,
        zorder=1,
    )
    ax.scatter(
        weeks,
        y,
        s=11,
        color="#222222",
        alpha=0.9,
        edgecolors="none",
        linewidths=0,
        label="Status quo",
        zorder=2,
    )
    ax.scatter(
        weeks,
        x,
        s=11,
        color="#8a8178",
        alpha=0.9,
        edgecolors="none",
        linewidths=0,
        label="Oracle",
        zorder=3,
    )
    ax.set_xlim(week_min - 1, week_max + 1)
    ax.set_xticks(week_ticks)
    ax.set_ylim(0, max_axis)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=8)


def plot_weekly_oracle_statusquo_minutes(
    paired_minutes: dict[str, np.ndarray],
    output_path: Path | str,
) -> None:
    """Two-panel weekly physical-minutes profile against the oracle."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(5.25, 3.8), sharex=True, constrained_layout=True)
    weeks = paired_minutes[WEEK_COLUMN]

    _plot_weekly_minutes_panel(
        axes[0],
        weeks,
        paired_minutes["realized_overtime_minutes__Oracle"],
        paired_minutes["realized_overtime_minutes__StatusQuo"],
        title="(a) Overtime",
        ylabel="Overtime (min/week)",
    )
    _plot_weekly_minutes_panel(
        axes[1],
        weeks,
        paired_minutes["realized_idle_minutes__Oracle"],
        paired_minutes["realized_idle_minutes__StatusQuo"],
        title="(b) Idle time",
        ylabel="Idle time (min/week)",
    )

    axes[0].set_xlabel("")
    axes[0].legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.align_ylabels(axes)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the .xlsx data file.")
    parser.add_argument("--sheet", default=None, help="Workbook sheet name. Defaults to the first sheet.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE1_PATH, help="Output path for Figure 1.")
    parser.add_argument(
        "--paired-weekly-data",
        type=Path,
        default=None,
        help=(
            "Path to paired_weekly_deltas.csv. Defaults to the newest compatible "
            "artifact under artifacts/experiments."
        ),
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=DEFAULT_FIGURE2_PATH,
        help="Output path for the Oracle/status-quo weekly physical-minutes figure.",
    )
    parser.add_argument(
        "--figure",
        choices=("all", "booking-deviation", "weekly-minutes"),
        default="all",
        help="Which paper figure(s) to generate.",
    )
    parser.add_argument("--threshold", type=float, default=30, help="Tolerance band half-width in minutes.")
    parser.add_argument("--x-clip", type=float, default=480, help="Displayed x-axis clipping limit in minutes.")
    parser.add_argument("--bin-width", type=float, default=5, help="Histogram bin width in minutes.")
    parser.add_argument(
        "--max-case-minutes",
        type=float,
        default=DEFAULT_MAX_PLANNING_CASE_MINUTES,
        help="Common planning filter: drop booked, room, or surgical durations above this value.",
    )
    args = parser.parse_args()

    if args.figure in {"all", "booking-deviation"}:
        diffs = load_deviations(
            args.data,
            args.sheet,
            max_case_minutes=args.max_case_minutes,
        )
        plot_booking_deviation(
            diffs,
            args.output,
            threshold=args.threshold,
            x_clip=args.x_clip,
            bin_width=args.bin_width,
        )

    if args.figure in {"all", "weekly-minutes"}:
        paired_weekly_path = args.paired_weekly_data or find_latest_paired_weekly_path()
        if paired_weekly_path is None:
            if args.figure == "weekly-minutes":
                raise SystemExit(
                    "No compatible paired_weekly_deltas.csv found. "
                    "Pass one with --paired-weekly-data."
                )
            print("Skipping weekly physical-minutes figure: no compatible paired_weekly_deltas.csv found.")
        else:
            paired_minutes = load_paired_weekly_minutes(paired_weekly_path)
            plot_weekly_oracle_statusquo_minutes(paired_minutes, args.paired_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
