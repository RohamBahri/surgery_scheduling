from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import final_paper_finalization_fixes as hardening
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def _arrays() -> base.Arrays:
    # bias, TWH dummy, extra case-level feature
    X = sparse.csr_matrix(
        [
            [1.0, 0.0, -0.86],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    booked = np.array([14.0, 100.0, 100.0])
    actual = np.array([3.6, 76.0, 76.0])
    return base.Arrays(
        X=X,
        booked=booked,
        actual=actual,
        error=actual - booked,
        week_ids=np.array([0, 0, 0]),
        case_ids=np.array([1, 2, 3]),
        week_slices={0: np.array([0, 1, 2])},
    )


def test_site_shift_fit_uses_deployment_clipping_and_is_valid() -> None:
    final.install_final_adapter()
    science.apply_scientific_fixes()
    a = _arrays()
    s = science.ScientificFinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    lam = 0.5

    class Enc:
        feature_names = ["bias", "site_TWH", "extra"]

    full = np.array([-30.0, 0.0, -19.767441860465116])
    w, hist = hardening.fit_site_shift_policy(a, Enc(), s, lam, full)
    assert w.shape == (3,)
    assert len(hist) == 2
    assert hist[-1]["deployment_clipping_is_part_of_fit_objective"] is True
    _, _, planning = base.correction_and_planning(w, a, s)
    assert np.all(np.isfinite(planning))
    assert np.all(planning > 0.0)

    # The deterministic fit must not be worse than the zero site-shift policy
    # under the same clipped case objective and regularizer.
    fitted = base.case_envelope(w, a, s) + lam * abs(w[1])
    zero = base.case_envelope(np.zeros(a.p), a, s)
    assert fitted <= zero + 1e-8


def test_guarded_vf_rejects_unattempted_search(tmp_path: Path, monkeypatch) -> None:
    a = _arrays()
    s = science.ScientificFinalSettings(data="dummy.xlsx", artifact_root=str(tmp_path))
    start = np.zeros(a.p)

    def fake_train_vf(*args, **kwargs):
        root = Path(args[7])
        (root / "VF_TRAJECTORY.csv").write_text("", encoding="utf-8")
        return np.asarray(args[0], dtype=float).copy(), []

    monkeypatch.setattr(hardening, "_ORIGINAL_TRAIN_VF", fake_train_vf)
    with pytest.raises(RuntimeError, match="before outer iteration 1"):
        hardening.guarded_train_vf(
            start,
            a,
            [],
            object(),
            {},
            s,
            1.0,
            tmp_path,
            search_deadline=0.0,
        )
    payload = json.loads((tmp_path / "VF_STATUS.json").read_text(encoding="utf-8"))
    assert payload["search_attempted"] is False
    assert payload["attempted_outer_iterations"] == 0


def test_finalization_version_is_frozen_string() -> None:
    assert hardening.FINALIZATION_FIXES_VERSION.startswith("final_paper_finalization_")
    assert hardening.EMERGENCY_WALL_TO_WORK_MULTIPLIER > 1.0
