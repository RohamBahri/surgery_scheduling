# Decision-aware operating-room planning

Research code for surgeon-mediated duration recommendations and elective operating-room planning. The retained method is **joint value-function (VF) learning via library MM/pDCA**, evaluated using the same deterministic, hash-tie-broken weekly MILP. CASE is a warm start and benchmark; VF learns all case and plan-feature coefficients together. Historical surgeon responses to recommendations are not observed; response parameters are **scenario assumptions**.

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

The study compares BOOKED, a response-unaware statistical recommendation, CASE, and joint VF. VF uses case features and congestion features from a bookings-based provisional plan. CASE initializes VF but does not constrain its coefficients. Fits use earlier cases; later outcomes are used for retrospective evaluation only. The 2013 evaluation period has already been examined during development and must **not** be presented as a fresh independent holdout.

## Focused joint-VF tuning

For a single behavioral scenario, train all case and plan-feature coefficients
jointly under the **saturating (monotone) response**. Display the canonical
advice and apply the original three-regime response in evaluation. The following
five-run check tests the reference penalty, weaker/stronger overall penalties,
stronger plan-feature penalties, and clustered score-based penalty weights:

```bash
python -u pilot/sweep_vf_tuning.py \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out-root artifacts/vf_tuning --cores 15 --plan-seconds 180
```

See `artifacts/vf_tuning/summary.csv`. Use `--only reference plan4` for an
initial smaller run. Each variant uses its own training directory. The private
context file and Gurobi are required. The reused 2013 outcomes make this
**exploratory tuning**, not an independent test.

## Privacy and scope

**No patient-level data, private context files, or results belong in Git.** Store the UHN workbook in a local `data/` directory and checkpoints, fitted coefficients, reports, and case-level outputs in local `artifacts/`; both are ignored. The repository does not include the private cohort-preparation checkpoint or a public replacement for its upstream preprocessing, so reproducing the study from a raw workbook alone requires the separately maintained private preparation pipeline.

Only the retained learning, planner, response evaluation, and focused tests are versioned. Joint VF requires fresh training artifacts; do not resume old five-coefficient VF checkpoints. Historical alternative methods, ray calibration, diagnostics, and feature prototypes are not part of the supported workflow.
