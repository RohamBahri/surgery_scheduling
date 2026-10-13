"""Recovery tests: validate a 225-cell grid before reusing any certificates."""
import json

import numpy as np
import pandas as pd
import pytest

from run_weekly_learning_recover import _cell_key, _read_previous


def previous_grid(tmp_path):
    manifest = {
        "train_weeks": 3, "validation_weeks": 2,
        "offsets": [-30., -15., 0., 15., 30.],
        "slopes": [-15., 0., 15.]
    }
    rows = []
    for group in ("TGH", "TWH-main", "TWH-day"):
        for period, weeks in (("train", 3), ("validation", 2)):
            for w in range(weeks):
                for offset in manifest["offsets"]:
                    for slope in manifest["slopes"]:
                        rows.append(dict(
                            group=group, monday=str(pd.Timestamp("2011-11-21")
                                + pd.Timedelta(days=7*w)),
                            period=period, offset=offset, slope=slope,
                            complete=True, cases=20,
                            case_error=15., vf=10., gap=5., spo=6.,
                            regret=2., realized_cost=22., oracle_cost=20.,
                            solver_seconds=.1, booked_moves=3,
                        ))
    # Training and validation periods must have disjoint week dates.
    for row in rows:
        if row["period"] == "validation":
            row["monday"] = str(pd.Timestamp(row["monday"]) + pd.Timedelta(days=21))
    rows[8]["complete"] = False
    for col in ("case_error", "vf", "gap", "spo", "regret",
                "realized_cost", "oracle_cost", "solver_seconds", "booked_moves"):
        rows[8][col] = np.nan
    path = tmp_path / "grid_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return manifest, path


def test_recover_grid_accepts_two_incomplete_rows(tmp_path):
    meta, path = previous_grid(tmp_path)
    df = _read_previous(tmp_path, meta)
    assert len(df) == 225
    assert len(df[~df.complete]) == 1
    key = _cell_key(df.loc[8].to_dict())
    assert key[0] == "TGH" and len(key) == 5


def test_recover_grid_rejects_invalid_completed_bounds(tmp_path):
    meta, path = previous_grid(tmp_path)
    grid = pd.read_csv(path)
    grid.loc[0, "gap"] = .5
    grid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="regret inequalities"):
        _read_previous(tmp_path, meta)


def test_recover_grid_rejects_duplicate_or_missing_cells(tmp_path):
    meta, path = previous_grid(tmp_path)
    grid = pd.read_csv(path)
    grid.loc[0, ["group", "monday", "period", "offset", "slope"]] = (
        grid.loc[1, ["group", "monday", "period", "offset", "slope"]].to_numpy()
    )
    grid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        _read_previous(tmp_path, meta)
    # Restore the unmodified grid before testing the independent missing-row
    # guard; the previous deliberate duplicate is a different error condition.
    _, path = previous_grid(tmp_path)
    grid = pd.read_csv(path).iloc[:-1]
    grid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected 225"):
        _read_previous(tmp_path, meta)
