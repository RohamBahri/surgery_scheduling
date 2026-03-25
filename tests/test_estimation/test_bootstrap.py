from src.core.config import Config
from src.core.types import Col
from src.estimation.bootstrap import SurgeonClusterBootstrap
from src.estimation.orchestrator import fit_estimation_pipeline


def _tiny_config() -> Config:
    cfg = Config()
    cfg.estimation.quantile_model.q_grid_size = 7
    cfg.estimation.inverse.n_min = 8
    cfg.estimation.response.n_folds = 2
    cfg.estimation.response.min_pairs = 4
    cfg.estimation.response.h_grid_max = 8.0
    cfg.estimation.response.h_grid_step = 4.0
    cfg.estimation.profile.n_profiles_per_service = 2
    cfg.estimation.bootstrap.n_bootstrap = 3
    cfg.estimation.bootstrap.q_grid_size_bootstrap = 5
    cfg.estimation.bootstrap.n_folds_bootstrap = 2
    cfg.estimation.bootstrap.n_jobs = 1
    cfg.estimation.bootstrap.random_seed = 7
    return cfg


def test_bootstrap_runs_on_tiny_synthetic_data(synthetic_df_train):
    cfg = _tiny_config()
    bs = SurgeonClusterBootstrap(cfg.estimation.bootstrap)
    out = bs.run(synthetic_df_train, cfg)

    assert len(out.q_hat_samples) > 0
    assert len(out.a_samples) > 0


def test_bootstrap_ci_returns_tuple(synthetic_df_train):
    cfg = _tiny_config()
    bs = SurgeonClusterBootstrap(cfg.estimation.bootstrap)
    out = bs.run(synthetic_df_train, cfg)

    surgeon = sorted(out.q_hat_samples.keys())[0]
    ci = out.ci(surgeon, "q_hat", alpha=0.1)
    assert isinstance(ci, tuple)
    assert len(ci) == 2


def test_duplicate_surgeon_copies_relabelled(synthetic_df_train):
    cfg = _tiny_config()
    bs = SurgeonClusterBootstrap(cfg.estimation.bootstrap)

    draw = ["S1", "S1", "S2"]
    df_boot = bs._resample_from_draw(synthetic_df_train, draw, iter_idx=2)
    labels = df_boot[Col.SURGEON_CODE].astype(str)

    assert any("S1__boot_0002__copy_01" == s for s in labels.unique())
    assert any("S1__boot_0002__copy_02" == s for s in labels.unique())


def test_duplicate_surgeon_copies_aggregated_not_overwritten(synthetic_df_train):
    cfg = _tiny_config()
    bs = SurgeonClusterBootstrap(cfg.estimation.bootstrap)
    out = bs._run_single_iteration(synthetic_df_train, cfg, iter_idx=1, seed=7)

    assert all("__boot_" not in k for k in out["q_hat"].keys())
    assert all("__boot_" not in k for k in out["a"].keys())


def test_profiles_not_bootstrapped(synthetic_df_train):
    cfg = _tiny_config()
    bs = SurgeonClusterBootstrap(cfg.estimation.bootstrap)
    _ = bs.run(synthetic_df_train, cfg)

    assert bs.saw_profiles_in_bootstrap is False


def test_orchestrator_attaches_bootstrap_when_requested(synthetic_df_train):
    cfg = _tiny_config()
    result = fit_estimation_pipeline(
        synthetic_df_train,
        cfg,
        skip_profiles=False,
        quiet=True,
        skip_bootstrap=False,
        run_bootstrap=True,
    )
    assert result.bootstrap is not None
