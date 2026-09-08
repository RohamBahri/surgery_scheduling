"""Generate paper-ready figures from the UHN surgery scheduling data.

Figure 1:
    Histogram of booking deviation, defined as realized room time minus booked
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
DEFAULT_FIGURE1_TRIMMED_PATH = DEFAULT_OUTPUT_DIR / "figure1_booking_deviation_hist_abs_le_240.pdf"
DEFAULT_FIGURE2_PATH = DEFAULT_OUTPUT_DIR / "figure2_oracle_statusquo_minutes.pdf"
DEFAULT_FIGURE1_TRIMMED_ABS_LIMIT = 240.0

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
    """Return realized room time minus booked time under the common filters."""
    rows, _ = load_filtered_analysis_rows(
        data_path,
        sheet,
        allow_order_violations=False,
        max_case_minutes=max_case_minutes,
    )
    return np.asarray([row.room_minutes - row.booked_minutes for row in rows], dtype=float)


def plot_booking_deviation(
    diffs: np.ndarray,
    output_path: Path | str,
    *,
    threshold: float = 30,
    x_clip: float = 480,
    bin_width: float = 5,
    figure_label: str = "Figure 1",
) -> None:
    """Histogram of realized room time minus booked time."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        raise ValueError("No finite booking deviations available for plotting.")

    mean_error = np.mean(diffs)
    mae = np.mean(np.abs(diffs))
    within = np.mean(np.abs(diffs) <= threshold) * 100
    overbooked = np.mean(diffs < 0) * 100
    diffs_display = np.clip(diffs, -x_clip, x_clip)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)

    ax.axvspan(-threshold, threshold, color="#f28e76", alpha=0.28, zorder=0)
    bins = np.arange(-x_clip, x_clip + bin_width, bin_width)
    ax.hist(
        diffs_display,
        bins=bins,
        color="#4a6fa5",
        edgecolor="white",
        linewidth=0.3,
        zorder=1,
    )
    ax.axvline(0, color="#009e73", linewidth=1.25, linestyle="-", zorder=2)

    ax.set_xlabel(r"Realized room time - booked time (minutes)")
    ax.set_ylabel("Number of cases")
    ax.set_xlim(-x_clip, x_clip)
    ax.set_xticks(np.arange(-x_clip, x_clip + 1, 60))

    annotation = (
        f"N = {n:,}\n"
        f"Mean = {mean_error:.0f} min\n"
        f"MAE = {mae:.0f} min\n"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7",
            linewidth=0.5,
        ),
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"{figure_label} summary:")
    print(f"  N: {n:,}")
    print(f"  Mean realized - booked error: {mean_error:.1f} min")
    print(f"  Percentage overbooked (booked > realized): {overbooked:.1f}%")
    print(f"  Percentage in shaded band (|error| <= {threshold:g} min): {within:.1f}%")
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


def _minute_axis_upper(*series: np.ndarray) -> float:
    data_min = min(float(np.min(values)) for values in series)
    data_max = max(float(np.max(values)) for values in series)
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
    panel_label: str,
    ylabel: str,
    y_upper: float,
) -> None:
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
        x,
        s=11,
        color="#3e7cb1",
        alpha=0.9,
        edgecolors="none",
        linewidths=0,
        label="Oracle",
        zorder=2,
    )
    ax.scatter(
        weeks,
        y,
        s=11,
        color="#c46a3a",
        alpha=0.9,
        edgecolors="none",
        linewidths=0,
        label="Status quo",
        zorder=3,
    )
    ax.set_xlim(week_min - 1, week_max + 1)
    ax.set_xticks(week_ticks)
    ax.set_ylim(0, y_upper)
    ax.set_xlabel("Week", labelpad=3)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="0.9", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.text(
        0.5,
        -0.34,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="top",
    )


def plot_weekly_oracle_statusquo_minutes(
    paired_minutes: dict[str, np.ndarray],
    output_path: Path | str,
) -> None:
    """Side-by-side weekly physical-minutes profile against the oracle."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    weeks = paired_minutes[WEEK_COLUMN]
    y_upper = _minute_axis_upper(
        paired_minutes["realized_overtime_minutes__Oracle"],
        paired_minutes["realized_overtime_minutes__StatusQuo"],
        paired_minutes["realized_idle_minutes__Oracle"],
        paired_minutes["realized_idle_minutes__StatusQuo"],
    )

    _plot_weekly_minutes_panel(
        axes[0],
        weeks,
        paired_minutes["realized_overtime_minutes__Oracle"],
        paired_minutes["realized_overtime_minutes__StatusQuo"],
        panel_label="(a) Overtime",
        ylabel="Minutes per week",
        y_upper=y_upper,
    )
    _plot_weekly_minutes_panel(
        axes[1],
        weeks,
        paired_minutes["realized_idle_minutes__Oracle"],
        paired_minutes["realized_idle_minutes__StatusQuo"],
        panel_label="(b) Idle time",
        ylabel="",
        y_upper=y_upper,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the .xlsx data file.")
    parser.add_argument("--sheet", default=None, help="Workbook sheet name. Defaults to the first sheet.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE1_PATH, help="Output path for Figure 1.")
    parser.add_argument(
        "--trimmed-output",
        type=Path,
        default=DEFAULT_FIGURE1_TRIMMED_PATH,
        help="Output path for the Figure 1 version filtered to bounded absolute errors.",
    )
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
        "--trimmed-abs-limit",
        type=float,
        default=DEFAULT_FIGURE1_TRIMMED_ABS_LIMIT,
        help="Absolute realized-booked error limit for the second Figure 1 version.",
    )
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
        finite_diffs = diffs[np.isfinite(diffs)]
        trimmed_diffs = finite_diffs[np.abs(finite_diffs) <= args.trimmed_abs_limit]
        removed = len(finite_diffs) - len(trimmed_diffs)
        print(
            f"Figure 1 trimmed version removes {removed:,} cases with "
            f"|realized - booked error| > {args.trimmed_abs_limit:g} min."
        )
        plot_booking_deviation(
            trimmed_diffs,
            args.trimmed_output,
            threshold=args.threshold,
            x_clip=min(args.x_clip, args.trimmed_abs_limit),
            bin_width=args.bin_width,
            figure_label=f"Figure 1 (|error| <= {args.trimmed_abs_limit:g} min)",
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
