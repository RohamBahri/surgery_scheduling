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

## Accuracy-constrained VF experiment (30-minute turnover, relaxed days)

This is the **current** experiment on branch `accuracy-constrained-vf-30min`.
The fixed-roster eligibility graph, 480-minute capacity, response model, and
83 policy features are unchanged. Observed surgeon weekdays remain metadata,
but the optimizer may empty an original operating day. Room-days remain
staffed and incur idle cost even when empty. No new surgeon workload cap or
extra eligible room/day is introduced.

The primary cost is followed lexicographically by minimum squared room loads
and deterministic hash. Training requests this **same** scheduling selection,
rather than using an arbitrary primary optimum. On timeout, the best feasible
assignment and certified primary lower bound are retained; the direct-MM
majorizer is accepted only if its improvement exceeds the aggregate gap.
Incomplete secondary/hash selections are explicitly counted; they are **not**
represented as certified deployment ties. A bounded quadratic-MIP uniqueness
test can skip hash selection when it proves the squared-load optimum unique.

The displayed-recommendation accuracy budget is

```text
A_display(theta) <= 1.05 * A_display(reference)
A_display = sum[1.75*max(actual - (booked + display), 0)
                + max((booked + display) - actual, 0)]
```

Both losses are averaged per training component. The reference is the
internally fitted prediction-only policy's **displayed advice**, not its
post-response reviewed booking. The display is the canonical recommendation
`ResponseDC.display(raw)`; only the proposed schedule uses the surgeon's
reviewed duration `ResponseDC.reviewed(raw)`. This models an accuracy
requirement on advice, not on hypothetical surgeon compliance.

The algorithm minimizes `VF + mu*A_display + lambda*L1`, using an exact DC
representation of the extra loss, and additionally checks the stated 5%
accuracy constraint at every accepted direct-MM iteration. A backtracking
proposal must satisfy both feasibility and the certified descent test.
The multiplier `mu=5` is a **predeclared computational choice**, not a
guaranteed optimal Lagrange multiplier: nonconvexity prevents claiming
equivalence with an exact constrained optimum. Check the final accuracy
ratio and whether updates were blocked by the budget.

### Run (fresh folder, private context, licensed Gurobi)

```bash
git switch main
git pull --ff-only
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q

mkdir -p artifacts/vf_accuracy_30min_relaxed
set -o pipefail

caffeinate -i python -u -m pilot.run_joint_vf \
  --stage train --turnover 30 \
  --accuracy-budget 0.05 --accuracy-mu 5 \
  --lambda-vf 0.1 --cores 15 --plan-seconds 600 \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_accuracy_30min_relaxed \
  2>&1 | tee artifacts/vf_accuracy_30min_relaxed/train.log
```

Once training completes, use exactly the same scientific and solver flags
with `--stage replay` for paired **training** scheduling costs.
Then use `--stage all --audit-seconds 120` to reuse the trained policy and
run the exploratory 2013 evaluation, probe, and overload audit. The model
configuration is frozen by `run_contract.json`; changing the multiplier,
budget, tie rules, turnover, or capacity requires a different output folder.

Important outputs: `vf_direct_progress.json` (descent, planner gaps, accuracy
and backtracking), `selected_policy.json` (budget, reference, achieved
accuracy, status), `training_replay_comparison.json` (paired training cost),
`summary.json` (common certified evaluation), `evaluation_2013.json`
(displayed and adopted error, surgeon-day workload), and
`operational_audit_summary.json` (forced vs avoidable overload).

The 2013 period has been reused in prior experiments and is exploratory,
not a pristine test for selecting `mu` or `lambda`.


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
  --out artifacts/vf_accuracy_30min_relaxed \
  --accuracy-mu 5 --accuracy-budget .05 \
  --turnover 30 --cores 15 --plan-seconds 600
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
| Scheduler | Fixed-roster eligibility MILP, 480-min room-days; experiment default 30-minute turnover |
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
- `hash-only`: frozen zero-turnover deployment comparison skipping squared-load
  balancing, retaining deterministic hash selection; no retraining.
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
The diagnostic branch \`vf-turnover-tie-diagnostics\` adds two explicit
options without changing the learned VF method.

**Frozen-policy hash-only check (no retraining, zero turnover):**

\`\`\`bash
git fetch origin
git switch vf-turnover-tie-diagnostics
git pull --ff-only
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage hash-only --turnover 0 \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_direct_no_turnover \
  --cores 15 --plan-seconds 300 \
  2>&1 | tee artifacts/vf_direct_no_turnover/hash_only.log
\`\`\`

The diagnostic strictly validates and reuses the original squared-load
deployment caches. It then solves the same frozen reviewed durations with
the squared-load stage skipped and the **same deterministic hash stages**.
No value is chosen using test realized durations. The output is
\`hash_only_diagnostic.json\`, with a paired comparison on instances for
which all three policies have proven deployments under both rules.

**One fresh 30-minute-turnover retraining experiment:**

\`\`\`bash
mkdir -p artifacts/vf_direct_turnover30
set -o pipefail
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage all --turnover 30 --lambda-vf 0.1 \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_direct_turnover30 \
  --cores 15 --plan-seconds 300 --audit-seconds 120 \
  2>&1 | tee artifacts/vf_direct_turnover30/run.log
\`\`\`

The 30-minute allowance is applied consistently to room loads, the
primary MILP, quadratic balancing, conservation shortcut, VF/DC/QP
identities, contextual features, and independent peak-load audit.
The turnover value is recorded in the run contract and included in
planner/audit cache signatures. All changed training artifacts must live
in a new \`--out\` directory. Never transplant saved VF coefficients or
feature scalers across these turnover settings.

Run \`--stage replay --turnover 30\` after completion, with the same
\`--out\` and \`--plan-seconds\`, for a paired training-cost comparison.
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
