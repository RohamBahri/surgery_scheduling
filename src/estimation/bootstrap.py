"""Surgeon-cluster bootstrap for estimation uncertainty."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.core.config import BootstrapConfig, Config
from src.core.types import Col, Domain


@dataclass
class BootstrapResult:
    q_hat_samples: Dict[str, list[float]] = field(default_factory=dict)
    a_samples: Dict[str, list[float]] = field(default_factory=dict)
    h_plus_samples: Dict[str, list[float]] = field(default_factory=dict)
    h_minus_samples: Dict[str, list[float]] = field(default_factory=dict)

    def ci(self, surgeon: str, param: str, alpha: float = 0.05) -> Tuple[float, float]:
        table = {
            "q_hat": self.q_hat_samples,
            "a": self.a_samples,
            "h_plus": self.h_plus_samples,
            "h_minus": self.h_minus_samples,
        }
        if param not in table:
            raise KeyError(f"Unsupported param={param}")
        vals = table[param].get(surgeon, [])
        if len(vals) == 0:
            return (float("nan"), float("nan"))
        lo = float(np.percentile(vals, 100 * (alpha / 2.0)))
        hi = float(np.percentile(vals, 100 * (1.0 - alpha / 2.0)))
        return lo, hi


class SurgeonClusterBootstrap:
    def __init__(self, config: BootstrapConfig | None = None) -> None:
        self.config = config or BootstrapConfig()
        self.saw_profiles_in_bootstrap = False

    def run(self, df_train: pd.DataFrame, pipeline_config: Config) -> BootstrapResult:
        n_boot = int(self.config.n_bootstrap)
        if n_boot <= 0:
            return BootstrapResult()

        seeds = [self.config.random_seed + i for i in range(n_boot)]
        if self.config.n_jobs > 1:
            outputs = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._run_single_iteration)(df_train, pipeline_config, i, seed)
                for i, seed in enumerate(seeds)
            )
        else:
            outputs = [
                self._run_single_iteration(df_train, pipeline_config, i, seed)
                for i, seed in enumerate(seeds)
            ]

        result = BootstrapResult()
        for out in outputs:
            for s, v in out["q_hat"].items():
                result.q_hat_samples.setdefault(s, []).append(float(v))
            for s, v in out["a"].items():
                result.a_samples.setdefault(s, []).append(float(v))
            for s, v in out["h_plus"].items():
                result.h_plus_samples.setdefault(s, []).append(float(v))
            for s, v in out["h_minus"].items():
                result.h_minus_samples.setdefault(s, []).append(float(v))
            self.saw_profiles_in_bootstrap = self.saw_profiles_in_bootstrap or bool(out["had_profiles"])
        return result

    def _run_single_iteration(
        self,
        df_train: pd.DataFrame,
        pipeline_config: Config,
        iter_idx: int,
        seed: int,
    ) -> dict:
        rng = np.random.default_rng(seed)
        base_surgeons = [s for s in sorted(df_train[Col.SURGEON_CODE].astype(str).unique()) if s != Domain.OTHER]
        if len(base_surgeons) == 0:
            return {"q_hat": {}, "a": {}, "h_plus": {}, "h_minus": {}, "had_profiles": False}

        sampled = rng.choice(base_surgeons, size=len(base_surgeons), replace=True).tolist()
        df_boot = self._resample_from_draw(df_train, sampled, iter_idx)

        cfg = deepcopy(pipeline_config)
        cfg.estimation.quantile_model.q_grid_size = self.config.q_grid_size_bootstrap
        cfg.estimation.response.n_folds = self.config.n_folds_bootstrap

        from src.estimation.orchestrator import fit_estimation_pipeline

        est = fit_estimation_pipeline(
            df_boot,
            cfg,
            skip_profiles=True,
            quiet=True,
            skip_bootstrap=True,
            run_bootstrap=False,
        )

        q_hat_lists: dict[str, list[float]] = defaultdict(list)
        for surgeon_code, q in est.critical_ratios.get_all_ratios().items():
            base = self._base_surgeon_code(surgeon_code)
            q_hat_lists[base].append(float(q))

        params_df = est.response_estimator.get_all_params()
        a_lists: dict[str, list[float]] = defaultdict(list)
        h_plus_lists: dict[str, list[float]] = defaultdict(list)
        h_minus_lists: dict[str, list[float]] = defaultdict(list)
        for _, row in params_df.iterrows():
            base = self._base_surgeon_code(str(row["surgeon_code"]))
            a_lists[base].append(float(row["a"]))
            h_plus_lists[base].append(float(row["h_plus"]))
            h_minus_lists[base].append(float(row["h_minus"]))

        q_hat = {s: float(np.mean(v)) for s, v in q_hat_lists.items()}
        a = {s: float(np.mean(v)) for s, v in a_lists.items()}
        h_plus = {s: float(np.mean(v)) for s, v in h_plus_lists.items()}
        h_minus = {s: float(np.mean(v)) for s, v in h_minus_lists.items()}

        return {
            "q_hat": q_hat,
            "a": a,
            "h_plus": h_plus,
            "h_minus": h_minus,
            "had_profiles": est.response_profiler is not None,
        }

    def _resample_from_draw(self, df_train: pd.DataFrame, sampled_surgeons: list[str], iter_idx: int) -> pd.DataFrame:
        copy_count: dict[str, int] = {}
        chunks: list[pd.DataFrame] = []

        for surgeon in sampled_surgeons:
            copy_count[surgeon] = copy_count.get(surgeon, 0) + 1
            suffix = f"__boot_{iter_idx:04d}__copy_{copy_count[surgeon]:02d}"
            new_code = f"{surgeon}{suffix}"
            part = df_train[df_train[Col.SURGEON_CODE].astype(str) == surgeon].copy()
            part[Col.SURGEON_CODE] = new_code
            chunks.append(part)

        other_rows = df_train[df_train[Col.SURGEON_CODE].astype(str) == Domain.OTHER].copy()
        if len(other_rows) > 0:
            chunks.append(other_rows)

        if len(chunks) == 0:
            out = df_train.iloc[0:0].copy()
        else:
            out = pd.concat(chunks, ignore_index=True)

        out = out.sort_values(Col.ACTUAL_START).reset_index(drop=True)
        out[Col.CASE_UID] = np.arange(len(out), dtype=int)
        return out

    def _base_surgeon_code(self, surgeon_code: str) -> str:
        marker = "__boot_"
        if marker in surgeon_code:
            return surgeon_code.split(marker)[0]
        return surgeon_code
