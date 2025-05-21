# stochastic_utils.py

import numpy as np
import pandas as pd

def build_empirical_distributions(df_warm: pd.DataFrame, params: dict):
    """
    Construct empirical duration distributions for each recoded
    main_procedure_id and a pooled fallback distribution.

    Expects df_warm with:
      - procedure_duration_min > 0
      - main_procedure_id (already recoded, e.g. "Other")
    Returns
    -------
    proc_samples : dict[str, np.ndarray]
        procedure_id → array of historical durations (min)
    all_samples  : np.ndarray
        pooled durations across all procedures
    """
    if df_warm.empty:
        print("[build_empirical_distributions] WARNING – warm-up empty.")
        return {}, np.empty(0, dtype=float)

    # keep only positive durations and clip to ≥ 1.0
    df = df_warm[df_warm["procedure_duration_min"] > 0].copy()
    df["procedure_duration_min"] = df["procedure_duration_min"].clip(lower=1.0)

    # group by recoded procedure id
    proc_groups = (
        df.groupby("main_procedure_id")["procedure_duration_min"]
          .apply(lambda arr: arr.to_numpy(dtype=float))
          .to_dict()
    )

    all_samples = df["procedure_duration_min"].to_numpy(dtype=float)

    print(
        f"[build_empirical_distributions] Built distributions for "
        f"{len(proc_groups)} procedures | total samples: {len(all_samples)}"
    )
    return proc_groups, all_samples


def sample_scenarios(surgeries: list, proc_samples: dict, all_samples: np.ndarray, params: dict):
    """
    Generate an (N × K) matrix of sampled durations for a list of surgeries.

    Each row i corresponds to surgery i in `surgeries`.
    Column k ∈ {0,…,K-1} is the sampled duration dᵢᵏ.

    Sampling rule:
      • If recoded proc_id has samples, draw with replacement from its array.
      • Else if all_samples nonempty, draw from pooled distribution.
      • Else fallback to booked_min for all scenarios.

    The RNG is seeded by params["saa_random_seed"] for reproducibility.
    """
    rng = np.random.default_rng(params.get("saa_random_seed", 42))
    K = params["saa_scenarios"]
    N = len(surgeries)

    if N == 0:
        return np.zeros((0, K), dtype=float)

    scen = np.zeros((N, K), dtype=float)

    for i, s in enumerate(surgeries):
        samples_i = proc_samples.get(s["proc_id"], None)

        if samples_i is None or len(samples_i) == 0:
            samples_i = all_samples
        if samples_i is None or len(samples_i) == 0:
            # fallback to booked time (≥ 1.0)
            scen[i, :] = max(1.0, float(s["booked_min"]))
        else:
            scen[i, :] = rng.choice(samples_i, size=K, replace=True)

    return scen
