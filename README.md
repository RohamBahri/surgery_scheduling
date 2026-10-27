# Decision-aware operating-room planning

Research code for surgeon-mediated duration recommendations and elective operating-room planning. The retained method is **CASE + value-function (VF) MM/pDCA**, evaluated using the same deterministic, hash-tie-broken weekly MILP. Historical surgeon responses to recommendations are not observed; response parameters are **scenario assumptions**.

## Install

Python 3.12 and a working Gurobi license are required for optimization.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## Reproduce the retained study

The input is a **private, locally generated** UHN context checkpoint; it is intentionally **not** in this repository:

`artifacts/objective_selection_12h/objective_selection_2013/context.pkl`

Run the four B-response training scenarios and frozen-advice evaluation:

```bash
python -u pilot/run_response_scenario_grid.py \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/response_saturation_grid --cores 15 --plan-seconds 180
```

Then evaluate the previously frozen advice under the additional C-response assumptions:

```bash
python -u pilot/extend_response_grid_evaluation.py \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/response_saturation_grid --cores 15 --plan-seconds 180
```

The study compares BOOKED, a response-unaware statistical recommendation, CASE, and VF. CASE and VF are fitted on earlier cases; later outcomes are used for retrospective evaluation only. The 2013 evaluation period has already been examined during development and must **not** be presented as a fresh independent holdout.

## Privacy and scope

**No patient-level data, private context files, or results belong in Git.** Store the UHN workbook in a local `data/` directory and checkpoints, fitted coefficients, reports, and case-level outputs in local `artifacts/`; both are ignored. The repository does not include the private cohort-preparation checkpoint or a public replacement for its upstream preprocessing, so reproducing the study from a raw workbook alone requires the separately maintained private preparation pipeline.

Only the retained learning, planner, response evaluation, and focused tests are versioned. Historical alternative methods, ray calibration, diagnostics, and feature prototypes are not part of the supported workflow.
