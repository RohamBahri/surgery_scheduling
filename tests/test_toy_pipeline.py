"""Optional small end-to-end pipeline smoke test, never uses private UHN data."""
import json
import pickle

import numpy as np
import pytest

from pilot import experiment
from surgery.weekly_data import Week


def gurobi_available():
    try:
        import gurobipy as gp
        m=gp.Model()
        m.Params.OutputFlag=0
        z=m.addVar()
        m.setObjective(z*z)
        m.optimize()
        m.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not gurobi_available(), reason="Gurobi license unavailable")
def test_one_scenario_complete_training_and_reporting(tmp_path):
    booked = np.array([140., 120.])
    actual = np.array([120., 135.])
    component = Week(
        "TGH", "2012-01-02", ((0, "OR1"), (0, "OR2")),
        np.array([10, 11]), ("S1",), np.array([0, 0]),
        np.array([0, 1]), ((0, 1),), ((0,),),
        booked, actual, ()).validate()
    source = dict(
        booked=booked, actual=actual, case_id=component.case_ids,
        X=np.array([[1., .5], [1., -.5]]))
    context = {
        "fit": source, "held": source,
        "H": np.zeros((2, 2)), "H_eval": np.zeros((2, 2)),
        "train": [("toy", component, np.arange(2))],
        "eval": [("toy", component, np.arange(2))],
    }
    checkpoint = tmp_path / "context.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump(context, handle)
    artifacts = tmp_path / "run"
    tuned = tmp_path / "selected_hyperparameters.json"
    tuned.write_text(json.dumps(dict(lambda_decision=.1, lambda_response=.1,
        candidates=[dict(method="DECISION_AWARE", lambda_value=.1,
                         scenario="saturating_a0p5_h30",
                         validation_weekly_cost=10., saving_vs_booked=2.)])))
    run_arguments = [
        "--context", str(checkpoint), "--out", str(artifacts),
        "--tuning", str(tuned),
        "--scenarios", "0.5:30", "--cores", "1",
        "--plan-seconds", "5", "--retry-seconds", "5",
        "--case-steps", "1", "--vf-outers", "1",
        "--vf-pdca-steps", "1", "--mu-candidates", "2",
    ]
    experiment.main(run_arguments)
    experiment.main(["--stage", "cross-evaluate", *run_arguments])
    assert (artifacts / "stress_summary.csv").is_file()
    assert (artifacts / "main_results.csv").is_file()
    assert (artifacts / "weekly_results.csv").is_file()
    from analysis.report import report
    output = artifacts / "publication"
    report(artifacts, checkpoint, output)
    assert (output / "numbers.json").is_file()
    assert (output / "figures" / "accuracy_vs_weekly_cost.svg").is_file()
    assert (output / "figures" / "lambda_validation.svg").is_file()
    assert (output / "figures" / "surgeon_response.svg").is_file()
