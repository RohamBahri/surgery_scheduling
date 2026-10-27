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

## Congestion representation ablation (preferred first test)

The case tables show that positive congestion reaches the existing +2
clipping threshold for about 7.5% of training cases. Before trying the broader
12-feature context model, test three *nested* plan feature sets using the
completed saturating-link `lambda025` run as reference:

- `congestion_booked`: original five plan columns plus centered
  `(booked/120 - 1) * congestion` (clipping stays ±2).
- `congestion_clip3`: original five columns with congestion clipped at ±3.
- `congestion_booked_clip3`: both changes together.

Run one variant at a time to isolate their scheduling effects:

```bash
python -u pilot/compare_vf_features.py \
  --base artifacts/vf_tuning/lambda025 \
  --vf-features congestion_booked \
  --out artifacts/vf_feature_comparison/congestion_booked \
  --cores 15 --plan-seconds 180
```

All fits use the **identical absolute L1 penalty** from the reference,
the original case initializer, the same provisional booked plans, and the
saturating training link with three-regime deployment response. Every case
and plan coefficient is optimized jointly. The runner saves
`vf_feature_diagnostics.json` to check for duplicates between base and
plan features; a near-zero scaler mean and unit standard deviation alone
do **not** establish duplication.

## Contextual booked-plan features (targeted diagnostic)

The original five booking-plan features remain available as
`--vf-features baseline` (default). An alternative
`--vf-features context` appends seven outcome-blind signals:
absolute congestion, proximity to 480-minute capacity within the
collectively attainable correction range, booked room case count, the case's
share of that room's booked minutes, number of alternative eligible slots,
and two interactions (congestion × share and congestion × flexibility).
All features use the frozen booked-duration provisional schedule; scaling
is fitted on training cases only. The complete case + plan coefficient
vector is still trained **jointly**.

To compare the richer features against a completed baseline with exactly
the same numeric L1 penalty (and the same nominal response), run:

```bash
python -u pilot/compare_vf_features.py \
  --base artifacts/vf_tuning/lambda025 \
  --out artifacts/vf_feature_comparison/context \
  --cores 15 --plan-seconds 180
```

Switch `--base` to a different completed saturating-link baseline if the
lambda sweep finds a better one. See `feature_comparison.json` in the
comparison folder. This uses the already-examined 2013 data and is
exploratory, not a pristine holdout.

## Unattended final-paper experiment

The research configuration is frozen: joint VF with the saturating training
link, original nominal (0.8,30) adoption, absolute L1 penalty
0.10924129431614196 and no additional feature search. This command
performs the complete empirical/methodological follow-up, with resumable
stage logs and source/checkpoint verification:

```bash
caffeinate -i python -u -m pilot.run_paper_completion \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --base artifacts/vf_tuning/lambda025 \
  --out artifacts/paper_completion \
  --cores 15 --plan-seconds 180
```

The script tests the installed Gurobi stack, audits saved numerical claims,
runs two controlled joint-VF fits (three nonduplicated plan features and a
*true* 77-coefficient case-only ablation), trains four homogeneous response
policies, evaluates frozen displayed advice across six deployment scenarios,
chooses nominal/minimax-regret/equal-weight candidate rules using *training*
outcomes, assesses two explicitly hypothetical persistent surgeon response
mixtures, and compares assignment hash-tie seeds 0–2.

Rerunning the **identical** command resumes cached planner solves and
completed stages; outputs appear in `artifacts/paper_completion/`. Start with
`final_report.md`, `final_results.json`, and
`audit/claims_register.csv`. The baseline policy is not overwritten.

The robust rules are chosen *from trained candidates*; no global
distributionally robust policy is claimed. No true surgeon adoption
parameters are identified; the 2013 data were examined repeatedly and the
results are exploratory. A proof obligation register is not a substitute for
completing each mathematical proof in the manuscript.

## Privacy and scope

**No patient-level data, private context files, or results belong in Git.** Store the UHN workbook in a local `data/` directory and checkpoints, fitted coefficients, reports, and case-level outputs in local `artifacts/`; both are ignored. The repository does not include the private cohort-preparation checkpoint or a public replacement for its upstream preprocessing, so reproducing the study from a raw workbook alone requires the separately maintained private preparation pipeline.

Only the retained learning, planner, response evaluation, and focused tests are versioned. Joint VF requires fresh training artifacts; do not resume old five-coefficient VF checkpoints. Historical alternative methods, ray calibration, diagnostics, and feature prototypes are not part of the supported workflow.
