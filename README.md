# Response-aware value-function learning for operating-room scheduling

This repository contains the **direct VF method and its main behavioral-grid experiment**.
A shared linear policy displays case-duration recommendations; a specified
surgeon response maps the displayed advice into the durations used by an
operating-room planner. The policy is trained with **direct MM/pDCA** of the
exact value-function surrogate and an accuracy safeguard. No finite schedule
library or workload-balancing secondary optimization is used.

The experiment is retrospective. Historical UHN records do **not** contain
actual responses to displayed recommendations. Results are conditional on
the stated behavioral model. The Jan–Jun 2013 evaluation has been examined
during development and is **exploratory, not a new untouched holdout**.

## Final scientific specification

| Component | Setting |
|---|---|
| Duration recommendation | Shared linear policy, 77 case + 6 structural features |
| Surgeon response | Saturating link; grid alpha in {0.5, 0.8}, h in {20, 30, 60} minutes; display bound +/-h |
| Display credibility | Displayed-duration MAE <= 1.05 times fitted accuracy-only reference |
| Accuracy penalty | mu=2 fixed in every scenario; report whether the 5% safeguard is satisfied |
| VF regularization | L1 lambda=0.1, intercept excluded |
| Learning algorithm | Exact direct-MM/pDCA, primary-optimal scheduling majorizers |
| Hospital roster | Fixed eligible room-days, capacity 480 minutes |
| Surgeon-day retention | Relaxed; observed original days kept only as metadata |
| Turnover in primary model | 0 minutes |
| Scheduling primary objective | Idle cost 1 and overtime cost 1.75 |
| Scheduling tie | **Deterministic hash only**, after optimal primary cost |
| Training | Jul 2011–Dec 2012, TGH/TWH |
| Evaluation | Jan–Jun 2013, TGH/TWH |

The fitted displayed-MAE reference is an internal calibration target, not a
second proposed scheduling method. The hard accuracy gate is off during
multiplier calibration; the selected fitted policy is checked against the
predeclared 5% budget on training data. This is a penalized solution with
verified feasibility, **not a globally certified constrained optimum**.

Training chooses any primary-optimal assignment as a valid majorizer.
Deployment uses the same primary objective followed by two deterministic
hash priorities and a canonical fallback if necessary. No squared-load
quadratic objective is solved. The planner logs both its feasible solutions
and optimization certifications. Uncertified instances are explicitly
excluded from comparisons labeled "common certified".

## Installation

Python 3.12 and a valid Gurobi license are required:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Private data and fitted artifacts are **not tracked** by Git. The context
checkpoint must exist locally at:

```text
artifacts/objective_selection_12h/objective_selection_2013/context.pkl
```

Do not delete this context or the original UHN workbook when cleaning local
ignored files.

## Main experiment: behavioral-scenario grid

There is **no designated main behavioral scenario**. The primary retrospective
experiment trains one independent **direct VF policy** for each combination of
the following saturating surgeon-response assumptions:

- Advice adoption alpha = 0.5 or 0.8.
- Credibility/saturation threshold h = 20, 30 or 60 minutes.

For each of these **six scenarios**, the comparison consists of exactly
four scheduling policies:

- **BOOKED:** plan directly on historical surgeon bookings.
- **NAIVE:** one shared 77-feature, regularized absolute-error duration predictor,
  trained without behavioral response or scheduling costs. Its predictions are
  displayed to the surgeon and passed through the same scenario's response.
- **VF:** the 83-feature response- and decision-aware policy, independently
  trained for that scenario using direct MM/pDCA and a fitted displayed-accuracy
  reference. The shared NAIVE coefficients are used only as the optimizer
  initialization; CASE is neither trained nor compared.
- **ORACLE:** a hindsight planner using realized surgery durations. It is a
  lower bound on realized scheduling cost when its plan is certified.

BOOKED and ORACLE schedules are solved once and reused across scenarios. The
training/test chronology, fixed roster, 480-minute capacity, zero turnover,
hash-only deployment tie rule, lambda=0.1, mu=2 and accuracy budget=5% remain
unchanged. Every policy uses the same test cases. A single primary grid reports
all four methods on **components certified for all four methods in all six
scenarios**; separate scenario-certified and all-feasible summaries remain
available. A failed 5% training accuracy safeguard is explicitly flagged; it is
not silently described as a feasible constrained solution.

**Run from repository root on the machine holding your private context:**

```bash
git pull --ff-only origin main
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q

caffeinate -i python -u -m pilot.run_vf_grid \
  --stage all \
  --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
  --out artifacts/vf_behavior_grid \
  --cores 15 --plan-seconds 300
```

For separate training and evaluation runs, use the identical command and
output directory with `--stage train`, then `--stage evaluate`.
Compatible completed policies and planner schedules are reused on restart.
Do not change the grid parameters or training settings in a populated output
directory. The optional `--alphas`, `--hs`, and `--links` arguments permit
**explicitly separately labeled** sensitivity grids (a changed grid requires a
new output directory).

**Primary files:**

- `grid.md`: readable six-scenario comparison using the common-certified test.
- `grid.csv`: all-feasible, scenario-certified and grid-common-certified
  costs, error measures, overtime, idle time and extreme-room-load counts.
- `grid_summary.json`: certification coverage, training accuracy checks,
  and paired calendar-week bootstrap intervals for VF minus BOOKED/NAIVE,
  on both scenario-specific and grid-common-certified samples.
- `component_results.csv`: complete component-level method outcomes and
  certification flags.
- `naive_lad.json`: the shared response-blind predictor.
- `scenarios/<scenario-id>/`: frozen VF coefficients, optimization history,
  accuracy reference, and scenario-specific evaluation details.

The 2013 evaluation has been inspected during prior development; this main
experiment gives a consistent final comparison, **not** an untouched
confirmatory holdout. No unrun grid result is claimed in the repository.

## Single-scenario diagnostic runner (not the main experiment)

After pulling the latest `main`, you can reuse an existing trained
`mu=2` policy if its scientific settings match. The final hash-only planner
cache has a new signature: previously computed **balanced** plans are not
silently reused, and hash-only deployments will be solved again as needed.

**Reproduce the paired training comparison:**

```bash
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage replay --accuracy-mu 2 --accuracy-budget .05 \
  --turnover 0 --train-tie primary --accuracy-gate off \
  --lambda-vf .1 --cores 15 --plan-seconds 300 \
  --out artifacts/vf_display_mae_mu2
```

**Reproduce the final exploratory evaluation and operational audit:**

```bash
caffeinate -i python -u -m pilot.run_joint_vf \
  --stage all --accuracy-mu 2 --accuracy-budget .05 \
  --turnover 0 --train-tie primary --accuracy-gate off \
  --lambda-vf .1 --cores 15 --plan-seconds 300 \
  --audit-seconds 120 \
  --out artifacts/vf_display_mae_mu2
```

The combined run fits VF only if a compatible frozen policy is absent.
It otherwise reuses the learned coefficients and re-solves the requested
hash-only deployment schedules. Save an archive of old results before
overwriting prior evaluation summaries.

**Core outputs:**

- `selected_policy.json`: coefficients, displayed-MAE reference and
  achieved training accuracy ratio.
- `vf_direct_progress.json`: optimizer convergence, verified penalized
  descent and planner gaps.
- `training_replay_comparison.json`: BOOKED versus VF paired training costs.
- `evaluation_2013.json` and `summary.json`: BOOKED versus VF realized
  2013 cost on a common certified sample, displayed MAE and workloads.
- `operational_audit_summary.json`: extreme loads and audited feasibility.

The previously used multiplier grid can still be reproduced with
`--stage train --accuracy-mu <value>`, using one distinct `--out`
directory for each candidate. Select using:

```bash
python -m pilot.select_accuracy_mu \
  artifacts/vf_display_mae_mu0.5 artifacts/vf_display_mae_mu1 \
  artifacts/vf_display_mae_mu2 artifacts/vf_display_mae_mu5 \
  --budget 0.05
```

Do not use the reused 2013 evaluation to revise the accuracy multiplier,
forecast features, scheduling rule or other scientific choices.

## Repository layout

- `pilot/run_joint_vf.py`: single public entry point.
- `pilot/_joint_vf_engine.py`: initialization, displayed-MAE reference,
  direct VF training, deployment evaluation, and reporting.
- `pilot/select_accuracy_mu.py`: training-only accuracy-budget calibration.
- `pilot/operational_audit.py`: feasibility and peak-load diagnostics.
- `surgery/`: data construction, response model, exact DC learning,
  structural features, and hash-only scheduling solver.
- `tests/`: mathematical identities, data contracts and regression checks.

No generated artifacts, private clinical data, old branches, or competing
experimental runners belong in the tracked tree.
