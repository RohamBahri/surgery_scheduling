"""Runner regression tests: deterministic component jobs and resumable results."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp

import numpy as np
import pytest

from run_weekly import _atomic_csv, _retry_rows, benchmark_component
from surgery.weekly_data import Week


def sample_component():
    return Week("TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
                np.array([2, 3, 4]), ("A", "B"), np.array([0, 0, 1]),
                np.array([0, 0, 1]), ((0, 1), (1,)), ((0,), (2,)),
                np.array([400., 200., 200.]), np.array([200., 400., 200.]),
                (("A", 2, "OR2", "prior_dates"),)).validate()


def test_spawn_workers_match_serial_exact_results():
    week = sample_component()
    key = dict(group="TGH", monday=week.monday, component=0, cases=3,
               surgeons=2, rooms=2, arcs=5, extra_days_offered=1,
               original_room_days=2)
    tasks = [(i, week, {**key, "component": i}, 30., 30., 2., 1, True)
             for i in (0, 1)]
    serial = [benchmark_component(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=2,
                             mp_context=mp.get_context("spawn")) as pool:
        parallel = list(pool.map(benchmark_component, tasks))
    for (_, expected), (_, result) in zip(serial, parallel):
        for field in ("booked_complete", "oracle_complete",
                      "adversary_complete", "booked_cost", "booked_regret",
                      "gap_vf_bound_at_booked", "spo_bound_at_booked"):
            assert result[field] == pytest.approx(expected[field])


def test_retry_only_failed_components(tmp_path):
    metadata = dict(dataset_sha256="fixed", prior_dates=1,
                    selected_group_weeks=2, selected_cases=6,
                    components=2, gamma=2., stage="benchmark",
                    adversary_enabled=True)
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    structure = [dict(group="TGH", monday="2012-01-02", component=i,
                      cases=3, surgeons=2, rooms=2, arcs=5,
                      extra_days_offered=1, original_room_days=2) for i in (0, 1)]
    records = {
        0: dict(**structure[0], booked_complete=True,
                oracle_complete=True, adversary_complete=True),
        1: dict(**structure[1], booked_complete=True,
                oracle_complete=True, adversary_complete=False),
    }
    csv = tmp_path / "benchmark.csv"
    _atomic_csv(records, csv)
    cached, pending = _retry_rows(csv, structure, metadata, adversary=True)
    assert list(cached) == [0] and pending == [1]
    bad = {**metadata, "prior_dates": 2}
    with pytest.raises(ValueError, match="prior_dates"):
        _retry_rows(csv, structure, bad, adversary=True)
