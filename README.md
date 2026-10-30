# Response-aware value-function learning for operating-room scheduling

This repository contains the **final single-method VF implementation**.
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
| Surgeon response | Saturating link, adoption alpha=0.8, display bound +/-30 min |
| Display credibility | Displayed-duration MAE <= 1.05 times fitted accuracy-only reference |
| Accuracy penalty | mu=2, selected on training data from {0.5, 1, 2, 5} |
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

## Run the final experiment

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
