"""Training-only multiplier selection never uses the 2013 evaluation."""
import json

import pytest

from pilot.select_accuracy_mu import select


def _run(folder,mu,ratio,reference=100.):
    folder.mkdir()
    (folder/"run_contract.json").write_text(json.dumps(dict(
        context="fixed",turnover=0.,relaxed_required_days=True,
        accuracy_mu=mu,accuracy_gate="off",
        accuracy_reference_kind="fitted_display_mae_reference",
        accuracy_kind="displayed_mae",accuracy_budget=.05)))
    (folder/"selected_policy.json").write_text(json.dumps(dict(
        accuracy_kind="displayed_mae",
        accuracy_reference_kind="fitted_display_mae_reference",
        accuracy_gate="off",accuracy_mu=mu,accuracy_budget=.05,
        accuracy_reference=reference,
        accuracy_achieved=reference*ratio,accuracy_ratio=ratio,
        vf_status="RELATIVE_IMPROVEMENT")))


def test_first_feasible_grid_multiplier(tmp_path):
    dirs=[tmp_path/x for x in ("one","two","three")]
    _run(dirs[0],2.,1.03)
    _run(dirs[1],.5,1.09)
    _run(dirs[2],1.,1.07)
    choice=select(dirs,budget=.05)
    assert choice["status"]=="SELECTED"
    assert choice["selected"]["mu"]==2.
    assert choice["candidates"][0]["mu"]==.5


def test_selector_rejects_reference_changes(tmp_path):
    first,second=tmp_path/"a",tmp_path/"b"
    _run(first,.5,1.1,reference=100.)
    _run(second,1.,1.02,reference=110.)
    with pytest.raises(ValueError,match="reference"):
        select([first,second],budget=.05)
