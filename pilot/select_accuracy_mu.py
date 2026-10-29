"""Select the smallest predeclared accuracy multiplier meeting a training MAE budget.

Usage:
 python -m pilot.select_accuracy_mu artifacts/vf_display_mae_mu* --budget .05

Selection uses only training-policy files; never touches the reused 2013 test.
"""
import argparse
import json
import math
from pathlib import Path


def select(directories, budget=.05):
    if not math.isfinite(budget) or budget<0:
        raise ValueError("Budget must be finite and nonnegative")
    reference_contract=None
    reference_mae=None
    records=[]
    seen=set()
    for root in directories:
        root=Path(root)
        policy=json.loads((root/"selected_policy.json").read_text())
        contract=json.loads((root/"run_contract.json").read_text())
        if (policy.get("accuracy_kind")!="displayed_mae" or
            policy.get("accuracy_reference_kind")!="fitted_display_mae_reference" or
            policy.get("accuracy_gate")!="off" or
            contract.get("accuracy_gate")!="off"):
            raise ValueError(f"Unexpected accuracy specification: {root}")
        mu=float(policy["accuracy_mu"])
        ratio=float(policy["accuracy_ratio"])
        ref=float(policy["accuracy_reference"])
        if (not all(math.isfinite(v) for v in (mu,ratio,ref)) or
            mu<0 or ref<=0 or mu in seen):
            raise ValueError("Invalid or duplicated multiplier/reference")
        seen.add(mu)
        if not math.isclose(float(policy["accuracy_budget"]),budget,abs_tol=1e-12):
            raise ValueError(f"Different budget in {root}")
        if not math.isclose(ratio,
                float(policy["accuracy_achieved"])/ref,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError(f"Accuracy ratio mismatch in {root}")
        scientific={k:v for k,v in contract.items()
                    if k not in ("accuracy_mu","source","source_history")}
        if reference_contract is None:
            reference_contract=scientific
            reference_mae=ref
        elif scientific!=reference_contract:
            raise ValueError("Scientific settings differ across multiplier runs")
        elif not math.isclose(ref,reference_mae,rel_tol=1e-8,abs_tol=1e-7):
            raise ValueError("Displayed-reference training losses differ")
        records.append(dict(folder=str(root),mu=mu,accuracy_ratio=ratio,
                  accuracy_met=bool(ratio<=1.+budget+1e-9),
                  vf_status=policy.get("vf_status")))
    records.sort(key=lambda x:x["mu"])
    feasible=[r for r in records if r["accuracy_met"]]
    return dict(budget=budget,reference_display_mae_per_component=reference_mae,
                candidates=records,selected=feasible[0] if feasible else None,
                status="SELECTED" if feasible else "NO_GRID_VALUE_MEETS_BUDGET")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders",nargs="+",type=Path)
    p.add_argument("--budget",type=float,default=.05)
    p.add_argument("--output",type=Path,default=None)
    a=p.parse_args()
    result=select(a.folders,a.budget)
    output=json.dumps(result,indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(output+"\n")
    print(output)
    if result["selected"] is None:
        p.exit(2,"No specified multiplier met the training accuracy budget.\n")


if __name__=="__main__":
    main()
