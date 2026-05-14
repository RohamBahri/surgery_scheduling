"""Targeted empirical analyses of booking behavior and booking signal.

This script applies the common cleaning rules used by the existing analysis
scripts, then produces two requested deliverables:

1. Self-correction null tests on consecutive surgeon/procedure pairs.
2. Fixed-effect regressions testing the marginal signal in booked time.

Usage:
    python3 scripts/targeter_empirical_analysis.py
    python3 scripts/targeter_empirical_analysis.py --data data/UHNOperating_RoomScheduling2011-2013.xlsx
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats as sp_stats

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / "artifacts" / ".matplotlib-cache"))
(CODE_ROOT / "artifacts" / ".matplotlib-cache").mkdir(parents=True, exist_ok=True)

from scripts.data_analysis import (  # noqa: E402
    MAX_PLANNING_CASE_MINUTES,
    add_derived_columns,
    load_data,
    preprocess,
)


DEFAULT_DATA = Path("data/UHNOperating_RoomScheduling2011-2013.xlsx")
DEFAULT_OUTPUT_DIR = Path("artifacts/targeter_empirical_analysis")
Z_975 = 1.959963984540054
MIN_OBSERVABILITY_DATE_COVERAGE = 0.50
MATERIAL_RESTRICTION_LOSS = 0.20


@dataclass
class OLSResult:
    nobs: int
    n_clusters: int
    params: pd.Series
    std_errors: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    pvalues: pd.Series


@dataclass
class FEModelResult:
    nobs: int
    n_clusters: int
    n_fe_cells: int
    within_r2: float
    mae: float
    params: pd.Series
    std_errors: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series


def clean_identifier(value: object) -> str | pd.NA:
    """Canonicalize IDs without applying any rare-category recoding."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return pd.NA
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "nat", "<na>"}:
        return pd.NA
    return text


def clean_patient_type(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = " ".join(str(value).replace("_", " ").replace("-", " ").split())
    if text == "" or text.lower() in {"nan", "none", "nat", "<na>"}:
        return pd.NA
    upper = text.upper()
    if upper == "INPATIENT":
        return "Inpatient"
    if upper == "SAME DAY PATIENT":
        return "Same Day Patient"
    if upper in {"EMERGENCY PATIENT", "EMERGENCY"}:
        return "EMERGENCY PATIENT"
    return text


def load_clean_cases(data_path: Path) -> pd.DataFrame:
    """Load, clean, and construct analysis columns requested in the spec."""
    required = [
        "Booked Time (Minutes)",
        "Enter Room Date",
        "Enter Room Time",
        "Leave Room Date",
        "Leave Room Time",
        "Actual Start Date",
        "Actual Start Time",
        "Actual Stop Date",
        "Actual Stop Time",
        "Surgeon_Code",
        "Main_Procedure_Id",
        "Patient_Type",
        "Operating_Room",
    ]
    raw = load_data(data_path)
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    raw = raw.copy()
    raw["__row_order"] = np.arange(len(raw), dtype=int)
    raw = add_derived_columns(raw)
    cleaned = preprocess(raw).copy()

    cleaned["booked"] = pd.to_numeric(cleaned["Booked Time (Minutes)"], errors="coerce")
    cleaned["realized"] = (
        (cleaned["Leave Room_DT"] - cleaned["Enter Room_DT"]).dt.total_seconds() / 60.0
    )
    cleaned["case_datetime"] = pd.to_datetime(cleaned["Enter Room_DT"], errors="coerce")
    cleaned["leave_room_date_only"] = pd.to_datetime(
        cleaned["Leave Room Date"], errors="coerce"
    ).dt.normalize()
    if "Decision_Date" in cleaned.columns:
        cleaned["decision_date_only"] = pd.to_datetime(
            cleaned["Decision_Date"], errors="coerce"
        ).dt.normalize()
    else:
        cleaned["decision_date_only"] = pd.NaT
    if "Consult_Date" in cleaned.columns:
        cleaned["consult_date_only"] = pd.to_datetime(
            cleaned["Consult_Date"], errors="coerce"
        ).dt.normalize()
    else:
        cleaned["consult_date_only"] = pd.NaT
    cleaned["error_signed"] = cleaned["realized"] - cleaned["booked"]
    cleaned["surgeon_id"] = cleaned["Surgeon_Code"].map(clean_identifier)
    cleaned["procedure_id"] = cleaned["Main_Procedure_Id"].map(clean_identifier)
    cleaned["patient_type_clean"] = cleaned["Patient_Type"].map(clean_patient_type)

    emergency = cleaned["patient_type_clean"].eq("EMERGENCY PATIENT")
    if emergency.any():
        cleaned = cleaned.loc[~emergency].copy()

    retained = {"Inpatient", "Same Day Patient"}
    observed = set(cleaned["patient_type_clean"].dropna().unique())
    unexpected = sorted(observed - retained)
    if unexpected:
        counts = cleaned.loc[
            cleaned["patient_type_clean"].isin(unexpected), "patient_type_clean"
        ].value_counts(dropna=False)
        raise SystemExit(
            "Unexpected Patient_Type values after standard cleaning. "
            "Spec retained only Inpatient, Same Day Patient, and missing:\n"
            + counts.to_string()
        )

    mask = (
        cleaned["booked"].notna()
        & cleaned["realized"].notna()
        & cleaned["case_datetime"].notna()
        & cleaned["surgeon_id"].notna()
        & cleaned["procedure_id"].notna()
        & (cleaned["booked"] > 0)
        & (cleaned["realized"] > 0)
        & (cleaned["realized"] <= MAX_PLANNING_CASE_MINUTES)
    )
    cleaned = cleaned.loc[mask].copy()
    cleaned["is_inpatient"] = cleaned["patient_type_clean"].eq("Inpatient").astype(float)
    cleaned["pt_missing"] = cleaned["patient_type_clean"].isna().astype(float)
    cleaned["patient_type_key"] = cleaned["patient_type_clean"].astype("object")
    cleaned.loc[cleaned["patient_type_key"].isna(), "patient_type_key"] = "(missing)"
    cleaned["surg_proc_fe"] = (
        cleaned["surgeon_id"].astype(str) + "||" + cleaned["procedure_id"].astype(str)
    )

    return cleaned.reset_index(drop=True)


def build_pair_dataset(
    df: pd.DataFrame,
    *,
    max_gap_days: int,
    same_procedure: bool = True,
    same_patient_type: bool = False,
) -> pd.DataFrame:
    group_cols = ["surgeon_id", "procedure_id"] if same_procedure else ["surgeon_id"]
    rows: list[dict[str, object]] = []
    ordered = df.sort_values(["surgeon_id", "procedure_id", "case_datetime", "__row_order"])

    for keys, group in ordered.groupby(group_cols, sort=False, dropna=False):
        group = group.sort_values(["case_datetime", "__row_order"]).reset_index(drop=True)
        if len(group) < 2:
            continue
        prev = group.shift(1)
        current = group
        gap_days = (
            current["case_datetime"] - prev["case_datetime"]
        ).dt.total_seconds() / 86400.0

        valid = gap_days.notna() & (gap_days > 0) & (gap_days <= max_gap_days)
        if same_patient_type:
            valid &= current["patient_type_key"].eq(prev["patient_type_key"])

        for idx in np.flatnonzero(valid.to_numpy()):
            curr = current.iloc[idx]
            old = prev.iloc[idx]
            e_prev = float(old["realized"] - old["booked"])
            rows.append(
                {
                    "surgeon_id": curr["surgeon_id"],
                    "procedure_id": curr["procedure_id"] if same_procedure else curr["procedure_id"],
                    "prev_procedure_id": old["procedure_id"],
                    "gap_days": float(gap_days.iloc[idx]),
                    "booked_curr": float(curr["booked"]),
                    "booked_prev": float(old["booked"]),
                    "delta_b": float(curr["booked"] - old["booked"]),
                    "e_prev": e_prev,
                    "e_pos": max(e_prev, 0.0),
                    "e_neg": min(e_prev, 0.0),
                    "is_inpatient": float(curr["is_inpatient"]),
                    "pt_missing": float(curr["pt_missing"]),
                    "leave_room_date_prev": old["leave_room_date_only"],
                    "decision_date_curr": curr["decision_date_only"],
                    "consult_date_curr": curr["consult_date_only"],
                    "patient_type_curr": curr["patient_type_key"],
                    "patient_type_prev": old["patient_type_key"],
                }
            )

    return pd.DataFrame(rows)


def fit_clustered_ols(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    cluster_col: str,
) -> OLSResult:
    needed = [y_col, cluster_col, *x_cols]
    work = data[needed].copy()
    for col in [y_col, *x_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=needed).copy()
    if len(work) <= len(x_cols) + 1:
        raise ValueError("Insufficient observations for OLS.")

    y = work[y_col].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(work)), work[x_cols].to_numpy(dtype=float)])
    names = ["Intercept", *x_cols]
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta

    clusters = work[cluster_col].astype(str).to_numpy()
    unique_clusters = pd.unique(clusters)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in unique_clusters:
        idx = clusters == cluster
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)

    n = len(work)
    k = x.shape[1]
    g = len(unique_clusters)
    scale = 1.0
    if g > 1 and n > k:
        scale = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tcrit = sp_stats.t.ppf(0.975, df=g - 1) if g > 1 else Z_975
    tvals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = (
        2 * sp_stats.t.sf(np.abs(tvals), df=g - 1)
        if g > 1
        else 2 * sp_stats.norm.sf(np.abs(tvals))
    )

    params = pd.Series(beta, index=names)
    std_errors = pd.Series(se, index=names)
    return OLSResult(
        nobs=n,
        n_clusters=g,
        params=params,
        std_errors=std_errors,
        ci_lower=params - tcrit * std_errors,
        ci_upper=params + tcrit * std_errors,
        pvalues=pd.Series(pvals, index=names),
    )


def fit_single_surgeon_ols(group: pd.DataFrame) -> tuple[float, float]:
    x = pd.to_numeric(group["e_prev"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group["delta_b"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(y)
    if n < 3 or np.var(x) <= 0:
        return np.nan, np.nan
    xmat = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(xmat.T @ xmat)
    beta = xtx_inv @ xmat.T @ y
    resid = y - xmat @ beta
    sigma2 = float((resid @ resid) / max(n - 2, 1))
    se = math.sqrt(float(sigma2 * xtx_inv[1, 1]))
    if se <= 0 or not math.isfinite(se):
        return float(beta[1]), np.nan
    tval = float(beta[1] / se)
    pval = float(2 * sp_stats.t.sf(abs(tval), df=n - 2))
    return float(beta[1]), pval


def benjamini_hochberg(pvalues: pd.Series, q: float = 0.05) -> pd.Series:
    p = pvalues.dropna().sort_values()
    significant = pd.Series(False, index=pvalues.index)
    m = len(p)
    if m == 0:
        return significant
    ranks = np.arange(1, m + 1)
    passed = p.to_numpy() <= (q * ranks / m)
    if not passed.any():
        return significant
    cutoff = p.iloc[np.flatnonzero(passed).max()]
    significant.loc[pvalues <= cutoff] = True
    return significant


def summarize_surgeon_level(pairs_60: pd.DataFrame, output_dir: Path) -> str:
    rows: list[dict[str, object]] = []
    pair_counts = pairs_60.groupby("surgeon_id").size()
    eligible = pair_counts[pair_counts >= 30].index
    for surgeon in eligible:
        sub = pairs_60.loc[pairs_60["surgeon_id"].eq(surgeon)]
        beta, pval = fit_single_surgeon_ols(sub)
        rows.append(
            {
                "surgeon_id": surgeon,
                "n_pairs": int(len(sub)),
                "beta": beta,
                "p_value": pval,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        summary = "of 0 surgeons with >= 30 pairs, 0 had FDR-significant beta > 0; 0 had beta < 0; median beta = NA, IQR [NA, NA]"
        (output_dir / "analysis1_surgeon_level.csv").write_text("", encoding="utf-8")
        return summary

    result["fdr_significant"] = benjamini_hochberg(result["p_value"], q=0.05)
    result.to_csv(output_dir / "analysis1_surgeon_level.csv", index=False)

    finite_beta = result["beta"].dropna()
    median = float(finite_beta.median()) if len(finite_beta) else np.nan
    q25 = float(finite_beta.quantile(0.25)) if len(finite_beta) else np.nan
    q75 = float(finite_beta.quantile(0.75)) if len(finite_beta) else np.nan
    sig_pos = int((result["fdr_significant"] & (result["beta"] > 0)).sum())
    sig_neg = int((result["fdr_significant"] & (result["beta"] < 0)).sum())
    return (
        f"of {len(result):,} surgeons with >= 30 pairs, {sig_pos:,} had "
        f"FDR-significant beta > 0; {sig_neg:,} had FDR-significant beta < 0; "
        f"median beta = {median:.4f}, IQR [{q25:.4f}, {q75:.4f}]"
    )


def add_ols_row(
    rows: list[dict[str, object]],
    *,
    specification: str,
    term: str,
    result: OLSResult,
    n_pairs: int,
    n_surgeons: int,
    note: str = "",
) -> None:
    rows.append(
        {
            "specification": specification,
            "term": term,
            "estimate": result.params[term],
            "cluster_robust_se": result.std_errors[term],
            "ci_lower": result.ci_lower[term],
            "ci_upper": result.ci_upper[term],
            "n_pairs": n_pairs,
            "n_surgeons": n_surgeons,
            "note": note,
        }
    )


def run_analysis_1(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    pairs_60 = build_pair_dataset(df, max_gap_days=60, same_procedure=True)

    primary = fit_clustered_ols(pairs_60, "delta_b", ["e_prev"], "surgeon_id")
    add_ols_row(
        rows,
        specification="1.2 primary same surgeon-procedure, gap <= 60",
        term="e_prev",
        result=primary,
        n_pairs=len(pairs_60),
        n_surgeons=pairs_60["surgeon_id"].nunique(),
    )

    asym = fit_clustered_ols(pairs_60, "delta_b", ["e_pos", "e_neg"], "surgeon_id")
    add_ols_row(
        rows,
        specification="1.3 asymmetric positive previous error",
        term="e_pos",
        result=asym,
        n_pairs=len(pairs_60),
        n_surgeons=pairs_60["surgeon_id"].nunique(),
    )
    add_ols_row(
        rows,
        specification="1.3 asymmetric negative previous error",
        term="e_neg",
        result=asym,
        n_pairs=len(pairs_60),
        n_surgeons=pairs_60["surgeon_id"].nunique(),
    )

    surgeon_summary = summarize_surgeon_level(pairs_60, output_dir)

    robustness_specs = [
        ("R1 gap <= 30", dict(max_gap_days=30, same_procedure=True, same_patient_type=False)),
        ("R2 gap <= 60 primary repeat", dict(max_gap_days=60, same_procedure=True, same_patient_type=False)),
        ("R3 gap <= 90", dict(max_gap_days=90, same_procedure=True, same_patient_type=False)),
        ("R4 gap <= 180", dict(max_gap_days=180, same_procedure=True, same_patient_type=False)),
        ("R5 same surgeon any procedure, gap <= 60", dict(max_gap_days=60, same_procedure=False, same_patient_type=False)),
        ("R6 same surgeon-procedure and patient type, gap <= 60", dict(max_gap_days=60, same_procedure=True, same_patient_type=True)),
    ]
    for label, kwargs in robustness_specs:
        pairs = build_pair_dataset(df, **kwargs)
        result = fit_clustered_ols(pairs, "delta_b", ["e_prev"], "surgeon_id")
        add_ols_row(
            rows,
            specification=f"1.5 {label}",
            term="e_prev",
            result=result,
            n_pairs=len(pairs),
            n_surgeons=pairs["surgeon_id"].nunique(),
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "analysis1_table.csv", index=False)
    write_text_table(
        table,
        output_dir / "analysis1_table.txt",
        footer="\nSurgeon-level 1.4 summary: " + surgeon_summary + "\n",
    )
    return table, surgeon_summary


def choose_observability_date(pairs_60: pd.DataFrame) -> tuple[str | None, float, str]:
    """Choose the 1.6 date field without guessing when both are sparse."""
    decision_coverage = float(pairs_60["decision_date_curr"].notna().mean())
    consult_coverage = float(pairs_60["consult_date_curr"].notna().mean())
    if decision_coverage >= MIN_OBSERVABILITY_DATE_COVERAGE:
        return (
            "decision_date_curr",
            decision_coverage,
            (
                f"Decision_Date used for 1.6; pair-level coverage "
                f"{decision_coverage:.1%} vs Consult_Date {consult_coverage:.1%}."
            ),
        )
    if consult_coverage >= MIN_OBSERVABILITY_DATE_COVERAGE:
        return (
            "consult_date_curr",
            consult_coverage,
            (
                f"Consult_Date used for 1.6 because Decision_Date pair-level "
                f"coverage was {decision_coverage:.1%}; Consult_Date coverage "
                f"{consult_coverage:.1%}."
            ),
        )
    return (
        None,
        max(decision_coverage, consult_coverage),
        (
            "1.6 restriction skipped because Decision_Date and Consult_Date "
            f"pair-level coverage were sparse ({decision_coverage:.1%} and "
            f"{consult_coverage:.1%})."
        ),
    )


def apply_observability_restriction(
    data: pd.DataFrame,
    date_col: str | None,
) -> pd.DataFrame:
    if date_col is None:
        return data.copy()
    restricted = data.loc[
        data[date_col].notna()
        & data["leave_room_date_prev"].notna()
        & (data[date_col] > data["leave_room_date_prev"])
    ].copy()
    return restricted


def build_triplet_dataset(
    df: pd.DataFrame,
    *,
    prev_gap_days: int = 60,
    next_gap_days: int = 60,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = df.sort_values(["surgeon_id", "procedure_id", "case_datetime", "__row_order"])
    for _, group in ordered.groupby(["surgeon_id", "procedure_id"], sort=False, dropna=False):
        group = group.sort_values(["case_datetime", "__row_order"]).reset_index(drop=True)
        if len(group) < 3:
            continue
        for idx in range(1, len(group) - 1):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            next_case = group.iloc[idx + 1]
            gap_prev = (curr["case_datetime"] - prev["case_datetime"]).total_seconds() / 86400.0
            gap_next = (
                next_case["case_datetime"] - curr["case_datetime"]
            ).total_seconds() / 86400.0
            if not (math.isfinite(gap_prev) and math.isfinite(gap_next)):
                continue
            if not (0 < gap_prev <= prev_gap_days and 0 < gap_next <= next_gap_days):
                continue
            e_prev = float(prev["realized"] - prev["booked"])
            e_next = float(next_case["realized"] - next_case["booked"])
            records.append(
                {
                    "surgeon_id": curr["surgeon_id"],
                    "procedure_id": curr["procedure_id"],
                    "gap_prev_days": float(gap_prev),
                    "gap_next_days": float(gap_next),
                    "booked_curr": float(curr["booked"]),
                    "booked_prev": float(prev["booked"]),
                    "delta_b": float(curr["booked"] - prev["booked"]),
                    "e_prev": e_prev,
                    "e_pos": max(e_prev, 0.0),
                    "e_neg": min(e_prev, 0.0),
                    "e_next": e_next,
                    "e_next_pos": max(e_next, 0.0),
                    "e_next_neg": min(e_next, 0.0),
                    "is_inpatient": float(curr["is_inpatient"]),
                    "pt_missing": float(curr["pt_missing"]),
                    "leave_room_date_prev": prev["leave_room_date_only"],
                    "decision_date_curr": curr["decision_date_only"],
                    "consult_date_curr": curr["consult_date_only"],
                }
            )
    return pd.DataFrame(records)


def add_confounding_rows(
    rows: list[dict[str, object]],
    *,
    spec: str,
    sample: str,
    result: OLSResult,
    terms: list[tuple[str, str]],
    n_obs: int,
    n_surgeons: int,
    note: str = "",
) -> None:
    for term, coef_name in terms:
        rows.append(
            {
                "Spec": spec,
                "Sample": sample,
                "Coef name": coef_name,
                "Estimate": result.params[term],
                "SE": result.std_errors[term],
                "CI lower": result.ci_lower[term],
                "CI upper": result.ci_upper[term],
                "n_obs": n_obs,
                "n_surgeons": n_surgeons,
                "note": note,
            }
        )


def run_pair_confounding_specs(
    rows: list[dict[str, object]],
    pairs: pd.DataFrame,
    *,
    sample_label: str,
    note: str = "",
) -> None:
    n_surgeons = int(pairs["surgeon_id"].nunique())
    sym = fit_clustered_ols(
        pairs,
        "booked_curr",
        ["booked_prev", "e_prev", "is_inpatient", "pt_missing"],
        "surgeon_id",
    )
    add_confounding_rows(
        rows,
        spec="1.7 symmetric level",
        sample=sample_label,
        result=sym,
        terms=[("booked_prev", "rho"), ("e_prev", "beta")],
        n_obs=len(pairs),
        n_surgeons=n_surgeons,
        note=note,
    )

    asym = fit_clustered_ols(
        pairs,
        "booked_curr",
        ["booked_prev", "e_pos", "e_neg", "is_inpatient", "pt_missing"],
        "surgeon_id",
    )
    add_confounding_rows(
        rows,
        spec="1.7 asymmetric level",
        sample=sample_label,
        result=asym,
        terms=[
            ("booked_prev", "rho"),
            ("e_pos", "beta_pos"),
            ("e_neg", "beta_neg"),
        ],
        n_obs=len(pairs),
        n_surgeons=n_surgeons,
        note=note,
    )


def run_triplet_confounding_specs(
    rows: list[dict[str, object]],
    triplets: pd.DataFrame,
    *,
    sample_label: str,
    note: str = "",
) -> None:
    n_surgeons = int(triplets["surgeon_id"].nunique())
    ref = fit_clustered_ols(triplets, "delta_b", ["e_prev"], "surgeon_id")
    add_confounding_rows(
        rows,
        spec="1.8a reference on triplets",
        sample=sample_label,
        result=ref,
        terms=[("e_prev", "beta_ref")],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )

    plac = fit_clustered_ols(triplets, "delta_b", ["e_next"], "surgeon_id")
    add_confounding_rows(
        rows,
        spec="1.8b future-shock placebo",
        sample=sample_label,
        result=plac,
        terms=[("e_next", "beta_plac")],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )

    joint = fit_clustered_ols(
        triplets, "delta_b", ["e_prev", "e_next"], "surgeon_id"
    )
    add_confounding_rows(
        rows,
        spec="1.8c joint previous and future shocks",
        sample=sample_label,
        result=joint,
        terms=[("e_prev", "beta"), ("e_next", "gamma")],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )

    joint_asym = fit_clustered_ols(
        triplets,
        "delta_b",
        ["e_pos", "e_neg", "e_next_pos", "e_next_neg"],
        "surgeon_id",
    )
    add_confounding_rows(
        rows,
        spec="1.8d asymmetric joint shocks",
        sample=sample_label,
        result=joint_asym,
        terms=[
            ("e_pos", "beta_pos"),
            ("e_neg", "beta_neg"),
            ("e_next_pos", "gamma_pos"),
            ("e_next_neg", "gamma_neg"),
        ],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )

    combined = fit_clustered_ols(
        triplets,
        "booked_curr",
        ["booked_prev", "e_prev", "e_next", "is_inpatient", "pt_missing"],
        "surgeon_id",
    )
    add_confounding_rows(
        rows,
        spec="1.9 combined persistence + placebo",
        sample=sample_label,
        result=combined,
        terms=[("booked_prev", "rho"), ("e_prev", "beta"), ("e_next", "gamma")],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )

    combined_asym = fit_clustered_ols(
        triplets,
        "booked_curr",
        [
            "booked_prev",
            "e_pos",
            "e_neg",
            "e_next_pos",
            "e_next_neg",
            "is_inpatient",
            "pt_missing",
        ],
        "surgeon_id",
    )
    add_confounding_rows(
        rows,
        spec="1.9 combined asymmetric",
        sample=sample_label,
        result=combined_asym,
        terms=[
            ("booked_prev", "rho"),
            ("e_pos", "beta_pos"),
            ("e_neg", "beta_neg"),
            ("e_next_pos", "gamma_pos"),
            ("e_next_neg", "gamma_neg"),
        ],
        n_obs=len(triplets),
        n_surgeons=n_surgeons,
        note=note,
    )


def run_triplet_window_specs(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    *,
    sample_label: str,
    date_col: str | None,
    include_unrestricted: bool,
    note: str = "",
) -> None:
    for label, next_window in [("W1", 30), ("W2", 60), ("W3", 90), ("W4", 180)]:
        triplets = build_triplet_dataset(df, prev_gap_days=60, next_gap_days=next_window)
        if not include_unrestricted:
            triplets = apply_observability_restriction(triplets, date_col)
        n_surgeons = int(triplets["surgeon_id"].nunique())
        result = fit_clustered_ols(
            triplets,
            "booked_curr",
            [
                "booked_prev",
                "e_pos",
                "e_neg",
                "e_next_pos",
                "e_next_neg",
                "is_inpatient",
                "pt_missing",
            ],
            "surgeon_id",
        )
        add_confounding_rows(
            rows,
            spec=f"1.10 {label} next gap <= {next_window}",
            sample=sample_label,
            result=result,
            terms=[
                ("e_pos", "beta_pos"),
                ("e_neg", "beta_neg"),
                ("e_next_pos", "gamma_pos"),
                ("e_next_neg", "gamma_neg"),
            ],
            n_obs=len(triplets),
            n_surgeons=n_surgeons,
            note=note,
        )


def run_analysis_1_confounding_fixes(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs_full = build_pair_dataset(df, max_gap_days=60, same_procedure=True)
    date_col, date_coverage, date_note = choose_observability_date(pairs_full)
    pairs_restricted = apply_observability_restriction(pairs_full, date_col)
    pair_loss = (
        1.0 - len(pairs_restricted) / len(pairs_full) if len(pairs_full) else np.nan
    )
    material = bool(np.isfinite(pair_loss) and pair_loss > MATERIAL_RESTRICTION_LOSS)

    triplets_full = build_triplet_dataset(df, prev_gap_days=60, next_gap_days=60)
    triplets_restricted = apply_observability_restriction(triplets_full, date_col)
    triplet_loss = (
        1.0 - len(triplets_restricted) / len(triplets_full)
        if len(triplets_full)
        else np.nan
    )

    sample_summary = pd.DataFrame(
        [
            {
                "metric": "observability_date_source",
                "value": date_col or "none",
                "note": date_note,
            },
            {
                "metric": "observability_date_pair_coverage",
                "value": date_coverage,
                "note": "",
            },
            {
                "metric": "pairs_before_1_6",
                "value": len(pairs_full),
                "note": "",
            },
            {
                "metric": "pairs_after_1_6",
                "value": len(pairs_restricted),
                "note": "",
            },
            {
                "metric": "pair_loss_fraction",
                "value": pair_loss,
                "note": "material > 0.20" if material else "",
            },
            {
                "metric": "triplets_before_1_6",
                "value": len(triplets_full),
                "note": "",
            },
            {
                "metric": "triplets_after_1_6",
                "value": len(triplets_restricted),
                "note": "",
            },
            {
                "metric": "triplet_loss_fraction",
                "value": triplet_loss,
                "note": "",
            },
        ]
    )

    rows: list[dict[str, object]] = []
    restricted_note = date_note
    if material:
        restricted_note += (
            f" 1.6 changed the pair sample materially: {len(pairs_full):,} "
            f"to {len(pairs_restricted):,} pairs ({pair_loss:.1%} lost)."
        )
        run_pair_confounding_specs(
            rows,
            pairs_full,
            sample_label="full 60-day pairs (unrestricted)",
            note="reported because 1.6 material-loss flag was triggered",
        )
        run_triplet_confounding_specs(
            rows,
            triplets_full,
            sample_label="full 60-day triplets (unrestricted)",
            note="reported because 1.6 material-loss flag was triggered",
        )
        run_triplet_window_specs(
            rows,
            df,
            sample_label="full triplets (unrestricted)",
            date_col=date_col,
            include_unrestricted=True,
            note="reported because 1.6 material-loss flag was triggered",
        )

    date_label = {
        "decision_date_curr": "Decision_Date",
        "consult_date_curr": "Consult_Date",
    }.get(date_col, "")
    active_pair_sample = (
        f"{date_label} restricted 60-day pairs" if date_col else "full 60-day pairs"
    )
    active_triplet_sample = (
        f"{date_label} restricted 60-day triplets" if date_col else "full 60-day triplets"
    )
    active_window_sample = (
        f"{date_label} restricted triplets" if date_col else "full triplets"
    )
    run_pair_confounding_specs(
        rows,
        pairs_restricted,
        sample_label=active_pair_sample,
        note=restricted_note,
    )
    run_triplet_confounding_specs(
        rows,
        triplets_restricted,
        sample_label=active_triplet_sample,
        note=restricted_note,
    )
    run_triplet_window_specs(
        rows,
        df,
        sample_label=active_window_sample,
        date_col=date_col,
        include_unrestricted=False,
        note=restricted_note,
    )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "analysis1_confounding_fixes_table.csv", index=False)
    write_text_table(table, output_dir / "analysis1_confounding_fixes_table.txt")
    sample_summary.to_csv(
        output_dir / "analysis1_confounding_fixes_sample_summary.csv", index=False
    )
    write_text_table(
        sample_summary,
        output_dir / "analysis1_confounding_fixes_sample_summary.txt",
    )
    return table, sample_summary


def make_panel_frame(
    data: pd.DataFrame,
    *,
    fe_col: str,
    x_cols: list[str],
    y_col: str = "realized",
    cluster_col: str = "surgeon_id",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    needed = list(dict.fromkeys([y_col, fe_col, cluster_col, *x_cols]))
    work = data[needed].copy()
    for col in [y_col, *x_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=needed).copy()

    fe_counts = work.groupby(fe_col, dropna=False).size()
    usable_fe = fe_counts[fe_counts >= 2].index
    work = work.loc[work[fe_col].isin(usable_fe)].copy()
    work["__panel_obs"] = work.groupby(fe_col, sort=False).cumcount()
    index = pd.MultiIndex.from_arrays(
        [work[fe_col].astype(str), work["__panel_obs"].astype(int)],
        names=[fe_col, "obs_id"],
    )
    y = pd.Series(work[y_col].to_numpy(dtype=float), index=index, name=y_col)
    x = pd.DataFrame(
        work[x_cols].to_numpy(dtype=float),
        columns=x_cols,
        index=index,
    )
    clusters = pd.Series(work[cluster_col].astype(str).to_numpy(), index=index, name=cluster_col)
    return work, y, x, clusters


def fit_panel_fe(
    data: pd.DataFrame,
    *,
    fe_col: str,
    x_cols: list[str],
    y_col: str = "realized",
    cluster_col: str = "surgeon_id",
) -> FEModelResult:
    work, y, x, clusters = make_panel_frame(
        data, fe_col=fe_col, x_cols=x_cols, y_col=y_col, cluster_col=cluster_col
    )
    if len(work) <= len(x_cols) + 1:
        raise ValueError(f"Insufficient observations for FE model with {fe_col}.")
    model = PanelOLS(
        y,
        x,
        entity_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(cov_type="clustered", clusters=clusters, debiased=True)

    conf = result.conf_int()
    params = result.params.copy()
    std_errors = result.std_errors.copy()
    ci_lower = conf.iloc[:, 0].rename("ci_lower")
    ci_upper = conf.iloc[:, 1].rename("ci_upper")
    residuals = result.resids.dropna()
    return FEModelResult(
        nobs=int(result.nobs),
        n_clusters=int(clusters.nunique()),
        n_fe_cells=int(work[fe_col].nunique()),
        within_r2=float(result.rsquared_within),
        mae=float(np.mean(np.abs(residuals.to_numpy(dtype=float)))),
        params=params,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def add_analysis2_row(
    rows: list[dict[str, object]],
    *,
    specification: str,
    term: str,
    result: FEModelResult,
    baseline: FEModelResult | None = None,
    note: str = "",
) -> None:
    has_term = term in result.params.index
    rows.append(
        {
            "specification": specification,
            "term": term if has_term else "",
            "estimate": result.params[term] if has_term else np.nan,
            "cluster_robust_se": result.std_errors[term] if has_term else np.nan,
            "ci_lower": result.ci_lower[term] if has_term else np.nan,
            "ci_upper": result.ci_upper[term] if has_term else np.nan,
            "within_r2": result.within_r2,
            "mae": result.mae,
            "delta_within_r2": (
                result.within_r2 - baseline.within_r2 if baseline is not None else np.nan
            ),
            "delta_mae": (baseline.mae - result.mae if baseline is not None else np.nan),
            "n": result.nobs,
            "n_surgeons": result.n_clusters,
            "n_fe_cells": result.n_fe_cells,
            "note": note,
        }
    )


def add_anesthetic_dummies(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = data.copy()
    if "Anaesthetic_Type_Given" not in work.columns:
        raise SystemExit("Anaesthetic_Type_Given column missing; cannot run robustness R5.")
    anae = work["Anaesthetic_Type_Given"].astype("object")
    anae = anae.where(anae.notna(), "(missing)")
    anae = anae.map(lambda value: " ".join(str(value).split()) or "(missing)")
    dummies = pd.get_dummies(anae, prefix="anaesthetic", dtype=float)
    dummy_cols = list(dummies.columns)
    work = pd.concat([work.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return work, dummy_cols


def run_booking_model_variant(
    data: pd.DataFrame,
    *,
    fe_col: str,
    extra_cols: list[str] | None = None,
) -> tuple[FEModelResult, FEModelResult]:
    extra_cols = extra_cols or []
    base_x = ["is_inpatient", "pt_missing", *extra_cols]
    booked_x = ["is_inpatient", "pt_missing", "booked", *extra_cols]
    baseline = fit_panel_fe(data, fe_col=fe_col, x_cols=base_x)
    with_booking = fit_panel_fe(data, fe_col=fe_col, x_cols=booked_x)
    return baseline, with_booking


def run_analysis_2(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    model1 = fit_panel_fe(df, fe_col="surg_proc_fe", x_cols=["is_inpatient", "pt_missing"])
    model2 = fit_panel_fe(
        df, fe_col="surg_proc_fe", x_cols=["is_inpatient", "pt_missing", "booked"]
    )
    add_analysis2_row(rows, specification="Model 1 FE only", term="", result=model1)
    add_analysis2_row(
        rows,
        specification="Model 2 / R1 primary FE + booking",
        term="booked",
        result=model2,
        baseline=model1,
    )

    proc_counts = df.groupby("procedure_id").size().sort_values(ascending=False)
    top_k = max(1, int(math.ceil(0.10 * len(proc_counts))))
    common_procedures = set(proc_counts.head(top_k).index)
    common_df = df.loc[df["procedure_id"].isin(common_procedures)].copy()
    common_base, common_booking = run_booking_model_variant(
        common_df, fe_col="surg_proc_fe"
    )
    add_analysis2_row(
        rows,
        specification="R2 top 10 percent procedures",
        term="booked",
        result=common_booking,
        baseline=common_base,
        note=f"top {top_k:,} of {len(proc_counts):,} procedures by case count",
    )

    surgeon_base, surgeon_booking = run_booking_model_variant(df, fe_col="surgeon_id")
    add_analysis2_row(
        rows,
        specification="R3 surgeon FE only",
        term="booked",
        result=surgeon_booking,
        baseline=surgeon_base,
    )

    proc_base, proc_booking = run_booking_model_variant(df, fe_col="procedure_id")
    add_analysis2_row(
        rows,
        specification="R4 procedure FE only",
        term="booked",
        result=proc_booking,
        baseline=proc_base,
    )

    anae_df, anae_cols = add_anesthetic_dummies(df)
    anae_base, anae_booking = run_booking_model_variant(
        anae_df, fe_col="surg_proc_fe", extra_cols=anae_cols
    )
    add_analysis2_row(
        rows,
        specification="R5 primary + Anaesthetic_Type_Given dummies",
        term="booked",
        result=anae_booking,
        baseline=anae_base,
        note="pending data-steward verification; column is provenance-uncertain and may be post-hoc",
    )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "analysis2_table.csv", index=False)
    write_text_table(table, output_dir / "analysis2_table.txt")
    return table


def format_float(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6f}"


def write_text_table(df: pd.DataFrame, path: Path, footer: str = "") -> None:
    display = df.copy()
    float_cols = [
        "estimate",
        "cluster_robust_se",
        "ci_lower",
        "ci_upper",
        "Estimate",
        "SE",
        "CI lower",
        "CI upper",
        "within_r2",
        "mae",
        "delta_within_r2",
        "delta_mae",
        "value",
    ]
    for col in float_cols:
        if col in display.columns:
            display[col] = display[col].map(format_float)
    text = display.to_string(index=False)
    path.write_text(text + footer, encoding="utf-8")


def write_cleaning_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = [
        ("cleaned_cases_used", len(df)),
        ("surgeons", df["surgeon_id"].nunique()),
        ("procedures", df["procedure_id"].nunique()),
        ("surgeon_procedure_cells", df["surg_proc_fe"].nunique()),
        ("missing_patient_type_cases", int(df["patient_type_clean"].isna().sum())),
        ("inpatient_cases", int(df["patient_type_clean"].eq("Inpatient").sum())),
        ("same_day_patient_cases", int(df["patient_type_clean"].eq("Same Day Patient").sum())),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        output_dir / "cleaned_sample_summary.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run targeter empirical analyses on the surgery dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input dataset path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and plaintext outputs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean_cases(args.data)
    write_cleaning_summary(df, args.output_dir)

    analysis1, surgeon_summary = run_analysis_1(df, args.output_dir)
    analysis1_confounding, confounding_sample_summary = run_analysis_1_confounding_fixes(
        df, args.output_dir
    )
    analysis2 = run_analysis_2(df, args.output_dir)

    print("\nTargeter empirical analysis complete.")
    print(f"Cleaned sample: {len(df):,} cases")
    print(f"Analysis 1 table: {args.output_dir / 'analysis1_table.csv'}")
    print(f"Analysis 1 text:  {args.output_dir / 'analysis1_table.txt'}")
    print(f"Analysis 1 surgeon-level summary: {surgeon_summary}")
    print(
        "Analysis 1 confounding fixes table: "
        f"{args.output_dir / 'analysis1_confounding_fixes_table.csv'}"
    )
    print(
        "Analysis 1 confounding fixes text:  "
        f"{args.output_dir / 'analysis1_confounding_fixes_table.txt'}"
    )
    print(
        "Analysis 1 confounding fixes sample summary: "
        f"{args.output_dir / 'analysis1_confounding_fixes_sample_summary.csv'}"
    )
    print(f"Analysis 2 table: {args.output_dir / 'analysis2_table.csv'}")
    print(f"Analysis 2 text:  {args.output_dir / 'analysis2_table.txt'}")
    print(
        "Rows written: "
        f"analysis1={len(analysis1):,}, "
        f"analysis1_confounding={len(analysis1_confounding):,}, "
        f"analysis2={len(analysis2):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
