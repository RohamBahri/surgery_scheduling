"""Fast, Gurobi-free checks of scenario costs and library selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from surgery.asdbs import mean_cost, selected, directions, scenario_bank
from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate


def component(monday, ids, actual):
    return Week("TGH", monday, ((0, "OR1"), (0, "OR2")),
                np.asarray(ids, int), ("S1", "S2"), np.asarray([0, 1], int),
                np.asarray([0, 1], int), ((0, 1), (0, 1)),
                ((0,), (0,)), np.asarray([200., 200.]),
                np.asarray(actual, float), ())


class ASDBSTests(unittest.TestCase):
    def test_scenario_cost_equals_planner(self):
        c = component("2012-01-02", [1, 2], [210., 260.])
        draws = np.asarray([[190., 215.], [250., 290.], [170., 190.]])
        for z in ([0, 0], [0, 1], [1, 1]):
            target = np.mean([evaluate(c, z, a)["cost"] for a in draws])
            self.assertAlmostEqual(mean_cost(c, z, draws), target, places=7)

    def test_library_selection_favors_primary_cost(self):
        c = component("2012-01-02", [1, 2], [210., 260.])
        options = [[0, 0], [0, 1]]
        pred = np.asarray([260., 260.])
        values = [evaluate(c, z, pred)["cost"] for z in options]
        self.assertEqual(selected(c, options, pred), int(np.argmin(values)))

    def test_directions_cover_every_feature(self):
        X = np.column_stack([np.ones(18), np.random.default_rng(7).normal(size=(18, 24))])
        names = ["intercept"] + [f"column_{i}" for i in range(24)]
        covered = set()
        for r in range(10):
            for label, v in directions(X, names, np.zeros(18), r):
                if label.startswith("feature:"):
                    covered.add(int(np.flatnonzero(v)[0]))
        self.assertEqual(covered, set(range(X.shape[1])))

    def test_case_metadata_and_resume_completed_bic(self):
        from pilot.run_asdbs import get_case
        grid = (.0005, .001, .002, .004, .008, .016, .032)
        data = {"X": np.ones((3, 2))}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(lambda_l1=lam, bic_proxy=float(i),
                         coefficients=[i, 0.],
                         asymmetric_loss_per_case=42.)
                    for i, lam in enumerate(sorted(grid, reverse=True))]
            (root / "case_bic_path.json").write_text(json.dumps(rows))
            with patch("pilot.run_asdbs.fold_lasso"), patch(
                    "pilot.run_asdbs.core.case") as fitted:
                w = get_case(root, data)
                fitted.assert_not_called()
            self.assertEqual(w.tolist(), [0., 0.])
            self.assertEqual(
                json.loads((root / "case_policy.json").read_text())["method"],
                "CASE_BIC_PILOT")

    def test_fresh_case_metadata_has_no_method_collision(self):
        from pilot.run_asdbs import get_case
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = {"X": np.ones((3, 2))}
            def save_policy(args, data, metadata):
                self.assertNotIn("method", metadata)
                (root / "case_policy.json").write_text(json.dumps(
                    {"response_link":"inverse","w":[0.,0.]}))
            with patch("pilot.run_asdbs.fold_lasso"), patch(
                    "pilot.run_asdbs.core.case", side_effect=save_policy):
                self.assertEqual(get_case(root, data).tolist(), [0., 0.])

    def test_scenarios_exclude_own_monday(self):
        a = component("2012-01-02", [1, 2], [100., 200.])
        b = component("2012-01-09", [3, 4], [300., 400.])
        items = [("a", a, np.array([0, 1])), ("b", b, np.array([2, 3]))]
        data = dict(X=np.ones((4, 1)), booked=np.full(4, 200.),
                    actual=np.array([100., 200., 300., 400.]))
        scenarios = scenario_bank(items, data, np.array([0.]), .8, 30., 31)
        self.assertTrue(np.allclose(scenarios["a"], [[300., 400.]] * 50))
        self.assertTrue(np.allclose(scenarios["b"], [[100., 200.]] * 50))


if __name__ == "__main__":
    unittest.main()
