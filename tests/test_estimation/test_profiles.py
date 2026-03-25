import numpy as np
import pandas as pd

from src.core.config import ProfileConfig
from src.core.types import Domain
from src.estimation.profiles import ResponseProfiler


def _params_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "surgeon_code": ["S1", "S2", "S3", "S4", "S5", Domain.OTHER],
            "a": [0.20, 0.22, 0.95, 0.40, 0.42, 0.30],
            "h_plus": [10.0, 11.0, 120.0, 20.0, 19.0, 15.0],
            "h_minus": [9.0, 10.0, 110.0, 18.0, 21.0, 15.0],
            "is_individual": [True, True, False, True, True, False],
        }
    )


def _services() -> dict[str, str]:
    return {
        "S1": "SvcA",
        "S2": "SvcA",
        "S3": "SvcA",  # pooled outlier in same service
        "S4": "SvcB",
        "S5": "SvcB",
        Domain.OTHER: Domain.OTHER,
    }


def test_profiles_created():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())

    assert profiler.n_profiles >= 3
    assert len(profiler.get_all_profiles()) == profiler.n_profiles


def test_pooled_surgeons_mapped_after_clustering():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())

    p_s1 = profiler.get_profile("S1")
    p_s3 = profiler.get_profile("S3")
    assert p_s1.service == "SvcA"
    assert p_s3.service == "SvcA"
    assert p_s1.a < 0.5  # pooled outlier (S3) does not define service center


def test_other_profile_exists():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())
    p_other = profiler.get_profile(Domain.OTHER)
    assert p_other.service == Domain.OTHER


def test_get_profile_id_unknown_returns_other():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())
    assert profiler.get_profile_id("unknown") == profiler.get_profile_id(Domain.OTHER)


def test_sos2_knots_ordered():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())
    pid = profiler.get_profile_id("S1")
    x, _ = profiler.get_sos2_knots(pid, L_ti=80.0, U_ti=220.0, b_ti=120.0)

    assert np.all(np.diff(x) >= -1e-9)


def test_sos2_values_match_piecewise_formula():
    profiler = ResponseProfiler(ProfileConfig(n_profiles_per_service=1)).fit(_params_df(), _services())
    pid = profiler.get_profile_id("S1")
    p = profiler.get_profile("S1")

    L_ti, U_ti, b_ti = 80.0, 220.0, 120.0
    x, y = profiler.get_sos2_knots(pid, L_ti=L_ti, U_ti=U_ti, b_ti=b_ti)

    hm = min(p.h_minus, max(0.0, b_ti - L_ti))
    hp = min(p.h_plus, max(0.0, U_ti - b_ti))
    expected = np.array(
        [
            p.a * ((L_ti - b_ti) + hm),
            0.0,
            0.0,
            0.0,
            p.a * ((U_ti - b_ti) - hp),
        ]
    )
    assert np.allclose(y, expected)
    assert np.isclose(x[1], -hm)
    assert np.isclose(x[3], hp)
