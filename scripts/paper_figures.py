"""Generate paper-ready figures from the UHN surgery scheduling data.

Figure 1:
    Histogram of booking deviation, defined as booked time minus realized room
    time.  The caption is set in LaTeX, not on the figure.

All figures in this script must use the same case filtering rules as
scripts.booking_realized_time_analysis.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the .xlsx data file.")
    parser.add_argument("--sheet", default=None, help="Workbook sheet name. Defaults to the first sheet.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE1_PATH, help="Output path for Figure 1.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
