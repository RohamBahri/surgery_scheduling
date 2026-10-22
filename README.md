# Surgery duration recommendations — VF research

**Current paper method:** value-function (VF) learning for surgeon-mediated, congestion-aware operating-room planning.

The current research uses historical elective surgeries from UHN (2011–2013), with **24,506 training** and **8,774 retrospective evaluation** cases. The study compares a five-feature shared VF recommendation policy against matched statistical (STAT), CASE, and BOOKED controls and a direct scheduling-cost (SHARED) benchmark. Surgeons' response parameters are **assumed scenarios**, not estimated responses to displayed advice.

The completed October 2026 model-selection and mechanism experiments are frozen. The **2013 cohort has been examined repeatedly**; subsequent analyses are retrospective diagnostics, not new independent holdouts.

## Current entrypoints

| Purpose | Entry |
|---|---|
| Main method-selection study (completed; reproduce only if needed) | `bash run_selection_12h.sh` |
| VF mechanism and response follow-up (completed; reproduce only if needed) | `bash run_vf_followups.sh` |
| Stage-level method selection | `pilot/run_objective_selection.py` |
| Stage-level VF follow-ups | `pilot/run_vf_followups.py` |

The current runner relies on `pilot/run_theory_full.py` and the `surgery/` implementation, especially `objective_selection.py`, `theory_exact.py`, `anchored_weekly_solver.py`, and `weekly_planner.py`. The *anchored* solver module provides the common exact physical-room planner and deterministic hash tie rule; its name does not mean ANCHORED is a selected learning method.

The original workbook belongs at `data/UHNOperating_RoomScheduling2011-2013.xlsx`. Do **not** commit raw data or local artifacts. Frozen run results remain under `artifacts/`, which is git-ignored.

## Install and test

Use Python with a valid Gurobi license for solver-backed and full experiments.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## Experiment provenance and outputs

- [Selection protocol and limitations](docs/paper_selection_12h.md).
- [VF follow-up design and interpretation](docs/vf_followups.md).
- Original method selection: `artifacts/objective_selection_12h/objective_selection_2013/`.
- Follow-up summaries: `artifacts/vf_followups/vf_followups/summary.json`.

Source-code SHA and study configuration are recorded in checkpoint contracts. A changed version of scientific code requires a **new artifact directory**, not reuse of incompatible training checkpoints.

## Historical code

Earlier daily-planning, ANCHORED, GAP, weekly-surrogate, and exact bilevel research remains in Git history and selected research modules. Those investigations **are not competing headline methods for the current VF paper**. Their documentation is historical and may describe different costs, planner/tie rules, or cohorts. Do not launch an old experiment merely because a historical runner exists.

For current manuscript work, preserve the distinction between the model-based VF upper bound, the solver's exact deployment status, and observed retrospective realized modeled costs. We have not established global optimality of the trained five-parameter VF coefficients or external-hospital causal benefits.
