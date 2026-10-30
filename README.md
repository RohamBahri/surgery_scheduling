# Response-aware value-function (VF) learning for operating-room scheduling

This repository contains **one retained learning method**: value-function
(VF) learning. It learns one shared linear recommendation policy
from **77 case features and six schedule-independent capacity/eligibility
features**. A specified hypothetical surgeon response produces reviewed
durations, and a fixed-roster Gurobi planner schedules them. All 83 policy
coefficients are optimized jointly using direct MM/pDCA of the exact VF
objective, without a finite schedule-library approximation. The
case-only response-aware CASE model is retained only as an initializer and
benchmark; BOOKED is the original-duration comparator.

The historical UHN data cannot identify actual responses to displayed advice.
All response assumptions and retrospective cost results must be described as
conditional. The January–June 2013 cohort was reused during development;
its comparisons are **exploratory, not a pristine confirmatory test**.

## Accuracy-budgeted VF experiment (relaxed surgeon-days)

The **primary specification uses zero turnover**, 480-minute fixed-roster
room-days, the existing surgeon eligibility graph and the **relaxed**
historical surgeon-day retention requirement. The nominal 30-minute
turnover option remains available as a *separate sensitivity analysis*.
With relaxed days, turnover counted as utilized time rewards consolidating
cases merely to replace unused time with extra changeovers, even if no
overtime is avoided. Zero turnover avoids that particular incentive.
Historical surgeon-days remain metadata; all rostered room-days incur idle
cost even if empty.

**Deployment is unchanged:** minimize overtime and idle, then minimize the
sum of squared predicted room loads, then apply deterministic hash ties.
Training uses a primary-optimal assignment by default for an inexpensive
and valid direct-MM majorizer. Use `--train-tie deployed` only if the
additional optimization time is justified. This does not assert that
the primary-value surrogate measures realized deployment regret.

### Credibility as displayed-duration accuracy

`displayed_duration = booked + ResponseDC.display(raw)` is the actual
canonical advice shown to the surgeon (under the assumed response model).
Neither the raw unconstrained score nor the reviewed booking is the
accuracy target.

```text
A_display(theta) = sum_i abs(actual_i - displayed_duration_i(theta)) / train_components
A_display(theta) <= 1.05 * A_display(reference)
```

A separate accuracy-only fit uses **the same 83 features** as VF and
minimizes this displayed MAE, starting from the casewise predictor.
Its coefficients, achieved accuracy, and optimizer status are saved in
`display_accuracy_reference.json`. This is a fitted local reference;
because the response is nonlinear, its global optimality is not claimed.

VF uses the exact DC Lagrangian
`VF(theta) + mu*A_display(theta) + 0.1*||theta_nonintercept||_1`.
The scheduling value and its bound retain their original asymmetric
overtime/idle cost parameters. The added absolute-error term is distinct.

To operationalize a **predeclared 5% budget**, fit the grid
**mu = {0.5, 1, 2, 5}** in separate directories, and select the **smallest**
mu whose achieved **TRAINING** `accuracy_ratio` is at most 1.05.
If none meets the budget, report that result and reconsider the grid
without consulting 2013. Set `--accuracy-gate off` for this grid;
`--accuracy-gate on` is a different, explicitly guarded algorithm and
would prevent the grid from measuring the multipliers' natural tradeoff.
Nonconvexity precludes claiming exact constrained global optimality.

### Commands

After installing dependencies and ensuring the private `context.pkl` is
available:

```bash
git switch main
git pull --ff-only
python -m pytest -q
set -o pipefail

for MU in 0.5 1 2 5; do
  mkdir -p "artifacts/vf_display_mae_mu$MU"
  caffeinate -i python -u -m pilot.run_joint_vf \
    --stage train \
    --turnover 0 --train-tie primary --accuracy-gate off \
    --accuracy-mu "$MU" --accuracy-budget 0.05 \
    --lambda-vf 0.1 --cores 15 --plan-seconds 300 \
    --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
    --out "artifacts/vf_display_mae_mu$MU" \
    2>&1 | tee "artifacts/vf_display_mae_mu$MU/train.log" || break
done

python -m pilot.select_accuracy_mu \
  artifacts/vf_display_mae_mu0.5 artifacts/vf_display_mae_mu1 \
  artifacts/vf_display_mae_mu2 artifacts/vf_display_mae_mu5 \
  --budget 0.05
```

For the selected multiplier/folder **only**, run `--stage replay` to
obtain paired training realized cost, then `--stage all
--audit-seconds 120` for exploratory 2013 deployment evaluation. Keep
the **same** `--turnover 0 --train-tie primary --accuracy-gate off
--accuracy-mu ... --accuracy-budget 0.05 --lambda-vf 0.1
--cores 15 --plan-seconds 300` flags.

Key artifacts: `display_accuracy_reference.json` (accuracy-only fitted
reference), `selected_policy.json` (`accuracy_ratio`, `train_tie`,
and turnover), `vf_direct_progress.json` (descent, planner bounds,
accuracy and optional gate), `training_replay_comparison.json` (paired
training realized costs), and `summary.json` and
`operational_audit_summary.json` (exploratory 2013).

## Install

Python 3.12 and a valid Gurobi license are required for scheduling and
training. The full dataset and precomputed context are **private** and never
committed to GitHub.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

The existing cohort checkpoint is required at:

```
artifacts/objective_selection_12h/objective_selection_2013/context.pkl
```

## One command, multiple configurations

All new runs use **one public entry point**:

```bash
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage all \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_display_mae_mu1 \
  --accuracy-mu 1 --accuracy-budget .05 \
  --turnover 0 --train-tie primary --accuracy-gate off \
  --cores 15 --plan-seconds 300
```

This fits CASE as a warm start, learns VF jointly, evaluates BOOKED,
CASE and VF on commonly proven-optimal 2013 planning instances, reports realized
training cost, and audits extreme planned and realized room-day loads. It reuses complete checkpoints/cached planner solves where possible.
Each distinct scientific configuration **must use its own `--out` folder**.
Old context.pkl is input data only; the training, planning, and tie caches in
the new folder must be generated afresh. The new audit outputs
`operational_audit_summary.json` and per-policy `operational_audit_*.json`.
Training progress and certified exact-VF improvements are logged in
`vf_direct_progress.json`. Failed planner certifications remain explicit.

Default frozen scientific configuration:

| Setting | Value |
|---|---|
| Feature schema | Structural only: 77 case + 6 contextual columns |
| Training response | Saturating, alpha=0.8, h=30 min |
| Response display cap / duration floor | 180 min / 1 min |
| VF L1 lambda | 0.1 per independent training instance; intercept excluded |
| Penalty | Uniform; case and structural coefficients trained jointly |
| MM steps / pDCA steps | 30 / 50 |
| DC-cut budget | 80 |
| Coefficient bounds | +/-100 |
| Scheduler | Fixed-roster eligibility MILP, 480-min room-days; default zero turnover; optional 30 |
| Planning weights | Idle=1, overtime=1.75 |
| Deployment rule | Primary optimum, minimum sum of squared room loads, deterministic hash |

**Same method, changed response configuration** (use a new output folder):

```bash
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage all --alpha 0.5 --h 20 --link saturating \
  --out artifacts/vf_direct_a50_h20 --cores 15 --plan-seconds 300
```

To change numerical iteration budgets or the VF penalty without introducing
another method, use `--lambda-vf`, `--vf-outers`, `--vf-pdca-steps`,
`--cuts`, `--radius`, `--cores`, or `--plan-seconds`.
Record the chosen configurations before evaluating them; do not choose settings
using the repeatedly examined 2013 outcomes.

### Stages

All stages use `python -u -m pilot.run_joint_vf` with `--stage`:

- `all`: train, evaluate, compute realized training cost, and run operational audit.
- `train`: train the joint VF policy only.
- `evaluate`: evaluate frozen BOOKED, CASE and VF on 2013, then report.
- `probe`: realized training cost for frozen BOOKED and VF.
- `replay`: **CASE versus VF training replay**, adapted from the user's
  standalone `check_joint_vf_train.py`. It verifies the private context hash,
  case order, recorded booked/actual durations and frozen response, then solves
  CASE and VF with identical deterministic ties. Outputs
  `training_replay_comparison.json`.
- `seeds`: explicitly reschedule frozen BOOKED/CASE/VF under hash seeds
  0,1,2,3,4; common certified cases and calendar-week paired bootstrap. Outputs
  `seed_robustness.json`. Other seeds: `--seeds 0 1 2`.
- `hash-only`: frozen zero-turnover **BOOKED vs VF** evaluation using
  the exact primary-cost optimum followed by deterministic hash, without
  squared-load optimization or VF retraining. It runs independently of
  previous balanced 2013 plans and writes `hash_only_vf_comparison.json`.
  When compatible balanced plans already exist, it also reports a matched,
  commonly certified balanced-vs-hash comparison.
- `audit`: audit BOOKED, CASE and VF test plans for peak load concentration.
  Only flagged instances (default >720 planned or realized minutes) trigger
  independent minimum-feasible-peak MILPs under reviewed durations (prospective
  comparison) and, separately, observed durations (hindsight only). The report
  distinguishes confirmed avoidable concentration, capacity shortfalls,
  eligibility bottlenecks, and unresolved optimization gaps.
  `all` runs this audit automatically. Use `--audit-seconds 120` and
  `--audit-threshold 720` to adjust the audit.
- `stress`: replay **the same frozen displayed advice**, but apply one
  alternative assumed true response before rescheduling, without refitting.
  Configured by `--true-alpha`, `--true-h`, `--true-link`. Outputs
  `stress_<link>_a<alpha>_h<h>.json`.

For example, on a completed new direct-VF result folder:

```bash
python -u -m pilot.run_joint_vf --stage replay --out artifacts/vf_direct_no_turnover
python -u -m pilot.run_joint_vf --stage seeds --out artifacts/vf_direct_no_turnover \
  --seeds 0 1 2 3 4 --cores 15 --plan-seconds 300
python -u -m pilot.run_joint_vf --stage stress --out artifacts/vf_direct_no_turnover \
  --true-alpha 0.5 --true-h 20 --true-link nonmonotone --plan-seconds 300
```

**Do not delete or overwrite old local artifact directories.** The earlier
published research comparisons remain in the frozen archives, including
`vf_structural.zip` and `structural_finalization.zip`. They are not regenerated
by the cleaned repository. Historical source code is recoverable from commit
`f14d9f140e6ff529319358838305a399f20475b2` (before cleanup), without
keeping redundant executable study drivers on the current branch.


## Targeted turnover and scheduling-selection diagnostics

Run the existing paired training replay before changing the code checkout.
The diagnostic branch `vf-turnover-tie-diagnostics` adds two explicit
options without changing the learned VF method.

**Frozen-policy hash-only check (no retraining, zero turnover):**

```bash
git fetch origin
git switch vf-turnover-tie-diagnostics
git pull --ff-only
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage hash-only --turnover 0 \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_direct_no_turnover \
  --cores 15 --plan-seconds 300 \
  2>&1 | tee artifacts/vf_direct_no_turnover/hash_only.log
```

The diagnostic strictly validates and reuses the original squared-load
deployment caches. It then solves the same frozen reviewed durations with
the squared-load stage skipped and the **same deterministic hash stages**.
No value is chosen using test realized durations. The output is
`hash_only_diagnostic.json`, with a paired comparison on instances for
which all three policies have proven deployments under both rules.

**One fresh 30-minute-turnover retraining experiment:**

```bash
mkdir -p artifacts/vf_direct_turnover30
set -o pipefail
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage all --turnover 30 --lambda-vf 0.1 \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_direct_turnover30 \
  --cores 15 --plan-seconds 300 --audit-seconds 120 \
  2>&1 | tee artifacts/vf_direct_turnover30/run.log
```

The 30-minute allowance is applied consistently to room loads, the
primary MILP, quadratic balancing, conservation shortcut, VF/DC/QP
identities, contextual features, and independent peak-load audit.
The turnover value is recorded in the run contract and included in
planner/audit cache signatures. All changed training artifacts must live
in a new `--out` directory. Never transplant saved VF coefficients or
feature scalers across these turnover settings.

Run `--stage replay --turnover 30` after completion, with the same
`--out` and `--plan-seconds`, for a paired training-cost comparison.
The 2013 evaluation remains exploratory and must not be used to choose
the turnover assumption or regularization parameter on performance alone.

## Active code layout

- `pilot/run_joint_vf.py`: **only documented workflow entry point**.
- `pilot/operational_audit.py`: minimum-feasible-peak MILP diagnostic with
  certified lower/upper bounds separating avoidable concentration from forced load.
- `pilot/_joint_vf_engine.py`: CASE initialization, frozen joint VF training,
  checkpointing, and primary evaluation.
- `pilot/_seed_solver.py`: explicit-seed certified planner audit helper.
- `surgery/vf_features.py`: only the six structural feature functions and
  training-fitted normalization.
- `surgery/nonmonotone_learning.py`: response/DC compiler and direct
  MM/pDCA with gap-aware exact-VF acceptance (no persistent libraries).
- `surgery/anchored_weekly_solver.py`, `weekly_planner.py`: zero-turnover,
  squared-load secondary selection and fixed-roster assignment model.
- `surgery/weekly_data.py`, `surgery/data.py`: unchanged historical cohort logic.
- `surgery/response_grid.py`: application of alternative true responses to
  frozen displays, with no re-canonicalization.

There is deliberately **no runnable feature sweep, provisional-plan feature
model, old paper-completion pipeline, or duplicate response-grid trainer** in
the retained code. CASE and BOOKED are baselines, not competing learning
architectures. Past exploratory result archives and git history retain the
record of method selection.

## Claims and limits

The exact VF objective bounds planned-value-based realized scheduling cost
(with primary planner tolerance accounted for). Each accepted MM step verifies
a majorizer decrease larger than the planner's primal-dual gap, giving a
positive lower bound on exact-VF descent; it does **not** certify a global
optimum or numerical stationarity of the learned nonconvex policy. A penalized
training surrogate is not itself a realized-cost or regret certificate.

Train/test comparisons are on commonly certified components. Whole calendar
weeks—not individual cases or seed outcomes—are used as bootstrap units.
Neither the behavioral response parameters nor roster availability at decision
time are identified from the historical dataset. The old code remains available
at the pre-cleanup commit for historical result reproduction.
