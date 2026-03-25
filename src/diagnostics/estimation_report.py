"""Lightweight diagnostics for estimation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.types import Col

DEFAULT_DIAG_DIR = Path("outputs/estimation_diagnostics")


def _ensure_dir(output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_critical_ratio_distribution(result, output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path:
    out = _ensure_dir(output_dir)
    ratios = result.critical_ratios.get_all_ratios()
    vals = np.array(list(ratios.values()), dtype=float)

    summary = pd.DataFrame(
        {
            "metric": ["count", "mean", "std", "min", "p25", "p50", "p75", "max"],
            "value": [
                len(vals),
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.min(vals)),
                float(np.percentile(vals, 25)),
                float(np.percentile(vals, 50)),
                float(np.percentile(vals, 75)),
                float(np.max(vals)),
            ],
        }
    )
    path = out / "critical_ratio_distribution.csv"
    summary.to_csv(path, index=False)
    return path


def export_response_parameters(result, output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path:
    out = _ensure_dir(output_dir)
    params = result.response_estimator.get_all_params()
    path = out / "response_parameters.csv"
    params.to_csv(path, index=False)
    return path


def run_specification_checks(result, output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path:
    out = _ensure_dir(output_dir)
    params = result.response_estimator.get_all_params()
    ratios = result.critical_ratios.get_all_ratios()

    ratio_vals = np.array(list(ratios.values()), dtype=float)
    checks = {
        "n_surgeons_with_ratio": len(ratios),
        "ratio_range_ok": bool(np.all((ratio_vals >= 0.01) & (ratio_vals <= 0.99))),
        "response_a_range_ok": bool(((params["a"] > 0.0) & (params["a"] <= 1.0)).all()),
        "response_h_nonnegative": bool(((params["h_plus"] >= 0.0) & (params["h_minus"] >= 0.0)).all()),
        "n_profiles": 0 if result.response_profiler is None else int(result.response_profiler.n_profiles),
    }

    path = out / "specification_checks.json"
    path.write_text(json.dumps(checks, indent=2))
    return path


def export_profile_summary(result, output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path | None:
    if result.response_profiler is None:
        return None
    out = _ensure_dir(output_dir)
    profiles = result.response_profiler.get_all_profiles()
    df = pd.DataFrame([p.__dict__ for p in profiles])
    path = out / "profile_summary.csv"
    df.to_csv(path, index=False)
    return path


def export_quantile_model_quality(
    result,
    df_train: pd.DataFrame,
    output_dir: Path | str = DEFAULT_DIAG_DIR,
) -> Path:
    out = _ensure_dir(output_dir)
    pred = result.quantile_model.predict(df_train, q=0.5)
    y = df_train[Col.PROCEDURE_DURATION].to_numpy(dtype=float)

    quality = pd.DataFrame(
        {
            "observed": y,
            "pred_q50": pred,
            "error": y - pred,
        }
    )
    path = out / "quantile_model_quality.csv"
    quality.to_csv(path, index=False)
    return path


def export_bootstrap_confidence_intervals(result, output_dir: Path | str = DEFAULT_DIAG_DIR) -> Path | None:
    if result.bootstrap is None:
        return None
    out = _ensure_dir(output_dir)

    surgeons = sorted(set(result.bootstrap.q_hat_samples) | set(result.bootstrap.a_samples) | set(result.bootstrap.h_plus_samples) | set(result.bootstrap.h_minus_samples))
    rows = []
    for s in surgeons:
        q_lo, q_hi = result.bootstrap.ci(s, "q_hat")
        a_lo, a_hi = result.bootstrap.ci(s, "a")
        hp_lo, hp_hi = result.bootstrap.ci(s, "h_plus")
        hm_lo, hm_hi = result.bootstrap.ci(s, "h_minus")
        rows.append({
            "surgeon_code": s,
            "q_hat_lo": q_lo,
            "q_hat_hi": q_hi,
            "a_lo": a_lo,
            "a_hi": a_hi,
            "h_plus_lo": hp_lo,
            "h_plus_hi": hp_hi,
            "h_minus_lo": hm_lo,
            "h_minus_hi": hm_hi,
        })

    path = out / "bootstrap_confidence_intervals.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# Backward-compatible aliases.
plot_critical_ratio_distribution = export_critical_ratio_distribution
plot_response_parameters = export_response_parameters
plot_profile_summary = export_profile_summary
plot_quantile_model_quality = export_quantile_model_quality
plot_bootstrap_confidence_intervals = export_bootstrap_confidence_intervals
