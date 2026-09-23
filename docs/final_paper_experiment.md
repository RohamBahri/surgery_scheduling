# Final paper experiment protocol

## Primary specification

The paper experiment uses the audited fixed-capacity weekly planner.

- Site: TGH, weekdays Monday-Friday.
- Split: 72 training weeks followed by the untouched 22-week holdout; the holdout begins 2013-01-28.
- Capacity: `regular_template`, estimated from the 72 training weeks only.
- Block length: 480 minutes; all allocated blocks are fixed and have zero activation cost.
- Assignment: every case is assigned exactly once; no deferral.
- Eligibility: raw service / same-site room compatibility observed in at least 3 distinct training weeks; same-site fallback only when the primary set is empty.
- Turnover: 30 minutes per transition, with `n-1` transitions in an occupied block.
- Costs: overtime 15/minute, idle 10/minute.
- Weekly optimization: reduced `Psi` MILP. Paper costs are reported on the equivalent `Phi` scale through `Phi = K + Psi`.
- Response: primary symmetric specification `alpha=0.8`, `h=30`.
- Methods: Booked, Naive prediction, RA, OS, VF, Oracle.

`observed_activity_proxy` is retrospective and is not a valid primary prospective capacity definition. `median_count_template` is retained as a capacity sensitivity specification rather than a primary model.

## Why keep RA, OS, and VF

They identify different pieces of the method rather than three redundant competitors.

- **RA** uses response-aware asymmetric case weights but does not optimize over schedule switching.
- **OS** adds the downstream cost geometry of a fixed realized-duration oracle schedule.
- **VF** optimizes over a library of schedule/value-function surfaces and therefore allows the active schedule to change with the policy.

The paper should therefore present VF as the full method and RA/OS as structural ablations. Removing them would make it difficult to determine whether any improvement comes from response-awareness, fixed-schedule operational structure, or schedule switching.

## Main run on macOS

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m pytest tests/test_planning/test_fixed_capacity.py tests/test_final_paper_experiment.py -q

caffeinate -i python run_final_paper_experiment.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_experiment \
  --cores 15 \
  --max-wall-minutes 720
```

The experiment writes the training policies before the holdout is materialized. The final holdout should be consumed only after the training diagnostics and numerical quality are accepted.

## Paper outputs

The central outputs are:

- `DATA_FREEZE.json` and `WEEK_SPLIT.csv`: immutable sample and split record.
- `ORACLE_TRAIN.csv` / `ORACLE_HOLDOUT.csv`: oracle incumbents and bounds.
- `RA_PI_METRICS.json`: cross-fitted RA exposure diagnostics.
- `START_DECOMPOSITION.csv`: initial surrogate comparison.
- `VF_TRAJECTORY.csv`: outer-loop library/certificate trajectory.
- `TRAIN_DECOMPOSITION.csv`: case-envelope, scheduling/value-function, regularization, and prediction diagnostics.
- `LIBRARY_SUMMARY.csv`: weekly schedule-surface counts.
- `SATURATION_SUMMARY.json`: empirical reachable-box library stress test.
- `FINAL_HOLDOUT_WEEKLY.csv`: week-level realized costs and regret brackets.
- `FINAL_HOLDOUT_SUMMARY.csv`: main paper method comparison.
- `FINAL_DECISION.json` and `REPORT.md`: numerical readiness and run summary.

## Sensitivity plan

Do not mix sensitivities into the primary holdout result. Run them as separately labelled experiments after the primary pipeline is frozen. The priority sensitivities are:

1. turnover 0 / 20 / 30 / 40 minutes;
2. `regular_template` versus `median_count_template` capacity;
3. behavioural response strength / distrust scenarios, reported as mechanisms rather than alternative data splits;
4. planner time/gap sensitivity where numerical brackets remain material.

For the behavioural analysis, report not only realized cost but also acceptance/decay/discard rates, implemented MAE, schedule-change fraction, and VF library growth. This directly shows when RA, OS, and VF separate and why.