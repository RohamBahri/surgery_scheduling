import subprocess
import sys

from src.core.config import Config
from src.diagnostics.estimation_report import (
    plot_critical_ratio_distribution,
    plot_profile_summary,
    plot_quantile_model_quality,
    plot_response_parameters,
    run_specification_checks,
)
from src.estimation.orchestrator import fit_estimation_pipeline
from src.estimation.serialization import load_estimation, save_estimation


def _small_config() -> Config:
    cfg = Config()
    cfg.estimation.quantile_model.q_grid_size = 9
    cfg.estimation.inverse.n_min = 15
    cfg.estimation.response.n_folds = 3
    cfg.estimation.response.min_pairs = 6
    cfg.estimation.response.h_grid_max = 12.0
    cfg.estimation.response.h_grid_step = 3.0
    cfg.estimation.profile.n_profiles_per_service = 2
    return cfg


def test_pipeline_runs_end_to_end(synthetic_df_train):
    cfg = _small_config()
    result = fit_estimation_pipeline(synthetic_df_train, cfg, skip_profiles=False, quiet=True, skip_bootstrap=True)

    assert result.quantile_model is not None
    assert result.critical_ratios is not None
    assert result.response_estimator is not None
    assert result.response_profiler is not None


def test_serialization_round_trip(synthetic_df_train, tmp_path):
    cfg = _small_config()
    result = fit_estimation_pipeline(synthetic_df_train, cfg, skip_profiles=False, quiet=True, skip_bootstrap=True)

    path = tmp_path / "estimation.joblib"
    save_estimation(result, path)
    loaded = load_estimation(path)

    assert loaded.response_estimator is not None
    assert loaded.critical_ratios is not None


def test_diagnostics_run(synthetic_df_train, tmp_path):
    cfg = _small_config()
    result = fit_estimation_pipeline(synthetic_df_train, cfg, skip_profiles=False, quiet=True, skip_bootstrap=True)

    p1 = plot_critical_ratio_distribution(result, tmp_path)
    p2 = plot_response_parameters(result, tmp_path)
    p3 = run_specification_checks(result, tmp_path)
    p4 = plot_profile_summary(result, tmp_path)
    p5 = plot_quantile_model_quality(result, synthetic_df_train, tmp_path)

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4 is not None and p4.exists()
    assert p5.exists()


def test_cli_help_smoke():
    proc = subprocess.run(
        [sys.executable, "scripts/run_estimation.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--skip-profiles" in proc.stdout
    assert "--no-save" in proc.stdout
