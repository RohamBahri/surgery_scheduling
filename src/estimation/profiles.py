"""Response profile clustering and SOS2 knot generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.core.config import ProfileConfig
from src.core.types import Domain


@dataclass
class ResponseProfile:
    profile_id: int
    a: float
    h_plus: float
    h_minus: float
    n_surgeons: int
    service: str


class ResponseProfiler:
    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()
        self._profiles: dict[int, ResponseProfile] = {}
        self._surgeon_to_profile: dict[str, int] = {}
        self._other_profile_id: int | None = None

    @property
    def n_profiles(self) -> int:
        return len(self._profiles)

    def fit(self, params_df: pd.DataFrame, surgeon_services: Dict[str, str]) -> "ResponseProfiler":
        required = {"surgeon_code", "a", "h_plus", "h_minus", "is_individual"}
        missing = required - set(params_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = params_df.copy()
        df["service"] = df["surgeon_code"].map(surgeon_services).fillna(Domain.OTHER)

        individual = df[df["is_individual"].astype(bool)].copy()
        pooled = df[~df["is_individual"].astype(bool)].copy()

        self._profiles = {}
        self._surgeon_to_profile = {}
        next_pid = 0

        services = sorted(df["service"].unique().tolist())
        n_ind = {s: int((individual["service"] == s).sum()) for s in services}
        k_map = self._allocate_service_profile_counts(services, n_ind)

        for service in services:
            svc_all = df[df["service"] == service]
            svc_ind = individual[individual["service"] == service]
            k = k_map[service]

            service_profile_ids: list[int] = []
            if k <= 1:
                if len(svc_ind) > 0:
                    center = svc_ind[["a", "h_plus", "h_minus"]].mean().to_numpy(dtype=float)
                else:
                    center = svc_all[["a", "h_plus", "h_minus"]].mean().to_numpy(dtype=float)
                pid = next_pid
                next_pid += 1
                self._profiles[pid] = ResponseProfile(pid, float(center[0]), float(center[1]), float(center[2]), 0, service)
                service_profile_ids = [pid]

                for surgeon in svc_ind["surgeon_code"].astype(str).tolist():
                    self._surgeon_to_profile[surgeon] = pid
            else:
                X = svc_ind[["a", "h_plus", "h_minus"]].to_numpy(dtype=float)
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                centers = km.cluster_centers_
                order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
                remap = {old: new for new, old in enumerate(order)}
                labels = np.array([remap[l] for l in labels], dtype=int)
                centers = centers[order]

                for c in range(k):
                    pid = next_pid
                    next_pid += 1
                    center = centers[c]
                    self._profiles[pid] = ResponseProfile(
                        profile_id=pid,
                        a=float(center[0]),
                        h_plus=float(center[1]),
                        h_minus=float(center[2]),
                        n_surgeons=0,
                        service=service,
                    )
                    service_profile_ids.append(pid)

                for surgeon, label in zip(svc_ind["surgeon_code"].astype(str).tolist(), labels):
                    self._surgeon_to_profile[surgeon] = service_profile_ids[int(label)]

            svc_profiles = [self._profiles[pid] for pid in service_profile_ids]
            svc_pooled = pooled[pooled["service"] == service]
            for _, row in svc_pooled.iterrows():
                vec = np.array([row["a"], row["h_plus"], row["h_minus"]], dtype=float)
                distances = [np.linalg.norm(vec - np.array([p.a, p.h_plus, p.h_minus])) for p in svc_profiles]
                best_local = int(np.argmin(distances))
                self._surgeon_to_profile[str(row["surgeon_code"])] = service_profile_ids[best_local]

        for surgeon, pid in self._surgeon_to_profile.items():
            p = self._profiles[pid]
            self._profiles[pid] = ResponseProfile(p.profile_id, p.a, p.h_plus, p.h_minus, p.n_surgeons + 1, p.service)

        if len(individual) > 0:
            other_center = individual[["a", "h_plus", "h_minus"]].mean().to_numpy(dtype=float)
        else:
            other_center = np.array([0.3, 15.0, 15.0], dtype=float)

        other_pid = next_pid
        self._other_profile_id = other_pid
        self._profiles[other_pid] = ResponseProfile(
            profile_id=other_pid,
            a=float(other_center[0]),
            h_plus=float(other_center[1]),
            h_minus=float(other_center[2]),
            n_surgeons=0,
            service=Domain.OTHER,
        )
        self._surgeon_to_profile[Domain.OTHER] = other_pid

        for s in params_df["surgeon_code"].astype(str).tolist():
            self._surgeon_to_profile.setdefault(s, other_pid)

        return self

    def _allocate_service_profile_counts(self, services: list[str], n_individual: dict[str, int]) -> dict[str, int]:
        k = {}
        max_allowed = {}
        for s in services:
            n = n_individual[s]
            if n == 0:
                k[s] = 1
                max_allowed[s] = 1
            elif n < self.config.n_profiles_per_service:
                k[s] = 1
                max_allowed[s] = n
            else:
                k[s] = min(self.config.n_profiles_per_service, n)
                max_allowed[s] = n

        min_total = max(1, self.config.min_profiles_total)
        max_total = max(min_total, self.config.max_profiles_total)

        total = sum(k.values())
        while total > max_total:
            candidates = [s for s in services if k[s] > 1]
            if not candidates:
                break
            s = sorted(candidates, key=lambda x: (n_individual[x], -k[x], x))[0]
            k[s] -= 1
            total -= 1

        total = sum(k.values())
        while total < min_total:
            candidates = [s for s in services if k[s] < max_allowed[s]]
            if not candidates:
                break
            s = sorted(candidates, key=lambda x: (-n_individual[x], k[x], x))[0]
            k[s] += 1
            total += 1

        return k

    def get_profile(self, surgeon_code: str) -> ResponseProfile:
        pid = self.get_profile_id(surgeon_code)
        return self._profiles[pid]

    def get_profile_id(self, surgeon_code: str) -> int:
        if self._other_profile_id is None:
            raise RuntimeError("ResponseProfiler must be fitted before access")
        return self._surgeon_to_profile.get(surgeon_code, self._other_profile_id)

    def get_all_profiles(self) -> List[ResponseProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    def get_sos2_knots(
        self, profile_id: int, L_ti: float, U_ti: float, b_ti: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if profile_id not in self._profiles:
            raise KeyError(f"Unknown profile_id={profile_id}")
        p = self._profiles[profile_id]

        left_range = max(0.0, b_ti - L_ti)
        right_range = max(0.0, U_ti - b_ti)
        h_minus = min(max(0.0, p.h_minus), left_range)
        h_plus = min(max(0.0, p.h_plus), right_range)

        x = np.array([L_ti - b_ti, -h_minus, 0.0, h_plus, U_ti - b_ti], dtype=float)
        y = np.array(
            [
                p.a * ((L_ti - b_ti) + h_minus),
                0.0,
                0.0,
                0.0,
                p.a * ((U_ti - b_ti) - h_plus),
            ],
            dtype=float,
        )
        return x, y
