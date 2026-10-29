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
  --out artifacts/vf_direct_no_turnover \
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
| Scheduler | Fixed-roster eligibility MILP, 480-min room-days, zero turnover |
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
