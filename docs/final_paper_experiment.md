# Final paper experiment protocol

## Primary specification

The paper experiment uses the audited fixed-capacity weekly planner and the two largest cleaned UHN sites.

- Sites: TGH and TWH, weekdays Monday-Friday.
- Site selection: verified from cleaned pre-holdout weekday data only. TWH and TGH must remain the two largest sites or the run aborts.
- Split: 72 training weeks followed by the untouched 22-week holdout; the holdout begins 2013-01-28.
- Learning: one shared recommendation policy is fit using cases from both sites.
- Features: 111-dimensional case-local schema. The feature-family budgets are retained from the approved TGH specification, but represented service/surgeon/procedure levels are selected by frequency from the pooled 72-week training sample only. Site contributes one TGH-vs-TWH dummy.
- Capacity: `regular_template`, estimated from the pooled 72 training weeks only. Each site retains its own room blocks.
- Block length: 480 minutes; all allocated blocks are fixed and have zero activation cost.
- Assignment: every case is assigned exactly once; no deferral and no cross-site assignment.
- Eligibility: raw service / same-site room compatibility observed in at least 3 distinct training weeks; same-site fallback only when the primary set is empty.
- Turnover: 30 minutes per transition, with `n-1` transitions in an occupied block.
- Costs: overtime 15/minute, idle 10/minute.
- Weekly optimization: reduced `Psi` MILP. Paper costs are reported on the equivalent `Phi` scale through `Phi = K + Psi`.
- Response: primary symmetric specification `alpha=0.8`, `h=30`.
- Methods: Booked, Naive prediction, RA, OS, VF, Oracle.

Under the frozen workbook and cleaning pipeline, the pooled split contains 20,519 training cases and 6,561 holdout cases. The 72-week training sample contains 11,295 TWH and 9,224 TGH cases. These counts are run-time assertions, not manually adjustable targets.

The sites are pooled for learning but not physically pooled for planning. A TGH case can use only TGH blocks and a TWH case only TWH blocks. Thus each weekly objective is the sum of the two independent site planning costs under the same learned policy.

`observed_activity_proxy` is retrospective and is not a valid primary prospective capacity definition. `median_count_template` is retained as a capacity sensitivity specification rather than a primary model.

## Why keep RA, OS, and VF

They identify different pieces of the method rather than three redundant competitors.

- **RA** uses response-aware asymmetric case weights but does not optimize over schedule switching.
- **OS** adds the downstream cost geometry of a fixed realized-duration oracle schedule.
- **VF** optimizes over a library of schedule/value-function surfaces and therefore allows the active schedule to change with the policy.

The paper should therefore present VF as the full method and RA/OS as structural ablations. Removing them would make it difficult to determine whether any improvement comes from response-awareness, fixed-schedule operational structure, or schedule switching.

## Staged run on macOS

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m pytest tests/test_planning/test_fixed_capacity.py tests/test_final_paper_experiment.py -q
```

### Stage 1: training only

Run this first:

```bash
caffeinate -i python run_final_paper_training.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_training_two_site \
  --cores 15 \
  --max-wall-minutes 720
```

A successful training-only run ends with `TRAINING_FREEZE.json` and `RUN_STATUS.json` containing `TRAINING_COMPLETE_HOLDOUT_LOCKED`. It does not run the holdout oracle or evaluate any learned policy on the 22 final weeks.

Review the training artifacts and numerical diagnostics before proceeding.

### Stage 2: final holdout

Only after the Stage-1 artifacts are accepted, run the complete frozen experiment once:

```bash
caffeinate -i python run_final_paper_experiment.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_experiment_two_site \
  --cores 15 \
  --max-wall-minutes 720
```

The full experiment repeats the frozen training pipeline and only then evaluates the final holdout. The Stage-1 results are a gate, not input to a differently configured Stage-2 model.

## What the run does

1. Loads and cleans the workbook using only pre-holdout observations for site imputation.
2. Verifies that TWH and TGH are the two largest cleaned pre-holdout weekday sites.
3. Applies the two-site weekday scope and reproduces the frozen 72/22 chronological split.
4. Verifies 20,519 training cases and 6,561 holdout cases.
5. Fits the pooled 111-dimensional feature encoder using the 72 training weeks only.
6. Fits the pooled regular-template capacity and same-site raw-service eligibility from the 72 training weeks only.
7. Builds each week with TGH and TWH blocks in the same mathematical instance but forbids every cross-site case-block edge.
8. Audits reduced `Psi` against direct `Phi` on training weeks.
9. Solves realized-duration training oracles and booked schedules.
10. Fits Naive, RA, OS and then VF using one shared two-site policy.
11. Writes policy, pDCA, library, decomposition and saturation diagnostics.
12. In the training-only entry point, exits before holdout optimization/evaluation.
13. In the full entry point, materializes the 22 holdout weeks only after policies are frozen, then evaluates Booked, Naive, RA, OS, VF and Oracle.

## Paper outputs

The central outputs are:

- `DATA_FREEZE.json` and `WEEK_SPLIT.csv`: immutable sample and split record.
- `FEATURE_MANIFEST.json`: training-only pooled feature schema, references and selected categorical levels.
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
4. pooled two-site policy versus separately estimated site-specific policies as a negative-transfer diagnostic;
5. planner time/gap sensitivity where numerical brackets remain material.

For the behavioural analysis, report not only realized cost but also acceptance/decay/discard rates, implemented MAE, schedule-change fraction, and VF library growth. This directly shows when RA, OS, and VF separate and why.
