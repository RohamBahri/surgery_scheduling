# Pilot and experiment runners

The **active manuscript is VF**, not the historical daily experiment, ANCHORED, or EG. See the repository root [README](../README.md).

## Current

- `run_objective_selection.py`: staged, frozen-policy VF-versus-benchmarks research comparison.
- `run_vf_followups.py`: VF slack, response robustness, tie diagnostics, ablations, conditional transfer and duration heterogeneity.
- `run_theory_full.py`: shared exact planner, scenario evaluation, policy features, and other core routines used by both.

Use the repository root `run_selection_12h.sh` and `run_vf_followups.sh` only for **deliberate reproduction** of completed studies. Never overwrite the original `artifacts/` files with scientifically different source/configuration. The 2013 retrospective evaluation was previously inspected during model selection and is not pristine external validation.

## Historical (kept for reproducibility)

The `run_fold_pilot.py`, `run_final_2013.py`, `run_advice_comparison.py`, `run_asdbs.py`, and related pilots are earlier experiments with different planning conventions and research questions. They are **not** the current recommended workflow. Their original explanations and proofs remain in historical Git commits and the `docs/` directory.

Tests can be run from the root:

```bash
python -m pytest -q
```
