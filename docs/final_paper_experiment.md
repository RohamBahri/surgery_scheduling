# Final paper experiment protocol

## Primary specification

The paper experiment uses the audited fixed-capacity weekly planner and the two largest cleaned UHN sites.

- Sites: TGH and TWH, weekdays Monday-Friday.
- Site selection: verified from cleaned pre-holdout weekday data only. TWH and TGH must remain the two largest sites or the run aborts.
- Split: 72 training weeks followed by the untouched 22-week holdout; the holdout begins 2013-01-28.
- Cohort duration rule: all canonical quality/cancellation/emergency rules are retained, but the 480-minute planning eligibility cap is applied only to booked duration. Positive realized room/surgical durations above 480 remain in the outcome and can generate overtime.
- Learning: one shared recommendation policy is fit using cases from both sites.
- Features: 107-dimensional case-local predecision schema: intercept, standardized booked duration, training-selected service/surgeon/procedure indicators, and one site indicator. Calendar features based on realized surgery date are not used.
- Capacity: `median_count_template`, estimated from the pooled 72 training weeks only. Each site retains its own room blocks.
- Block length: 480 minutes; all allocated blocks are fixed and have zero activation cost.
- Assignment: every case is assigned exactly once; no deferral and no cross-site assignment. Cases may move across weekdays within the planning week.
- Eligibility: raw service / same-site room compatibility observed in at least 3 distinct training weeks; same-site fallback only when the primary set is empty.
- Turnover: 30 minutes per transition, with `n-1` transitions in an occupied block.
- Costs: overtime 15/minute, idle 10/minute.
- Weekly optimization: reduced `Psi` MILP. Paper costs are reported on the equivalent `Phi` scale through `Phi = K + Psi`; relative solver gaps are reported on the native `Psi` scale.
- Response: primary symmetric specification `alpha=0.8`, `h=30`.
- Coefficient box: +/-100, with the stronger case-level display/safety constraints retained.
- L1 regularization: method-specific relative scaling, `lambda_m = eta * L_m(0) / p`, `eta=0.01`.
- VF library: TGH and TWH schedule surfaces are stored separately; the value function takes the independent site minima, equivalent to the implicit Cartesian product under exact site separability.
- Deployment tie rule: one single-threaded fixed-seed solver pass with fixed predecision model ordering. There is no second tie-break MIP and no realized-outcome secondary objective.
- Methods: Booked, Naive prediction, RA, OS, VF, Oracle.

Under the frozen workbook and booked-only duration rule, the cleaned cohort contains 32,397 cases. The two-site weekday scope contains 30,363 cases. The final split contains 21,033 training cases and 6,713 holdout cases. Training contains 11,394 TWH and 9,639 TGH cases; holdout contains 3,849 TWH and 2,864 TGH cases. These counts are run-time assertions.

The sites are pooled for learning but not physically pooled for planning. A TGH case can use only TGH blocks and a TWH case only TWH blocks. Thus each weekly objective is the sum of two independent site planning costs under the same learned policy.

`observed_activity_proxy` remains retrospective-only. `regular_template` is retained as a capacity sensitivity/appendix specification rather than the primary model.

## Why keep RA, OS, and VF

They identify different pieces of the method rather than three redundant competitors.

- **RA** uses response-aware asymmetric case weights but does not optimize over schedule switching.
- **OS** adds the downstream cost geometry of a fixed realized-duration oracle schedule.
- **VF** optimizes over a library of schedule/value-function surfaces and therefore allows the active schedule to change with the policy.

VF is the full method; RA and OS are structural comparators. The manuscript should not describe RA-to-OS as a perfectly isolated one-factor ablation because the case-loss weighting also changes.

## Staged run on macOS

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m pytest \
  tests/test_planning/test_fixed_capacity.py \
  tests/test_final_paper_experiment.py \
  tests/test_final_paper_runtime_fixes.py \
  tests/test_final_paper_scientific_fixes.py \
  -q
```

### Stage 1: training only

Run this first with a fresh artifact directory:

```bash
caffeinate -i python run_final_paper_training.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_training_two_site \
  --cores 15 \
  --max-wall-minutes 720
```

A successful training-only run ends with `TRAINING_FREEZE.json` and `RUN_STATUS.json` containing `TRAINING_COMPLETE_HOLDOUT_LOCKED`. It also fingerprints the policy file, feature manifest, data freeze, frozen settings, scientific specification, and method-specific regularization record. It does not run the holdout oracle or evaluate a learned policy on the 22 final weeks.

Review the training artifacts and numerical diagnostics before proceeding.

### Stage 2: evaluation only

Only after accepting Stage 1, evaluate exactly that frozen bundle once:

```bash
caffeinate -i python run_final_paper_evaluation.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --training-artifact-root artifacts/final_paper_training_two_site \
  --artifact-root artifacts/final_paper_holdout_two_site
```

Stage 2 verifies the Stage-1 hashes, code commit, workbook hash, scientific-spec version, and feature/policy dimensions. It then loads the accepted `POLICIES.npz`; it never retrains. After successful evaluation it writes `HOLDOUT_EVALUATED.json` into the training bundle so accidental second consumption is refused.

Do not use `run_final_vf_experiment.py` or the old direct full-pipeline entry point for the final paper experiment.

## What the run does

1. Loads the workbook with canonical quality rules and the booked-only <=480-minute planning eligibility rule.
2. Verifies TWH and TGH as the two largest cleaned pre-holdout weekday sites.
3. Applies the two-site weekday scope and reproduces the frozen 72/22 chronological split.
4. Verifies 21,033 training cases and 6,713 holdout cases.
5. Fits the pooled 107-dimensional predecision feature encoder using training only.
6. Fits median-count capacity and same-site raw-service eligibility from the 72 training weeks only.
7. Builds day-flexible weekly instances with physically separate TGH/TWH capacity.
8. Audits reduced `Psi` against direct `Phi` on training weeks.
9. Solves realized-duration training oracles and booked schedules.
10. Fits Naive, RA, OS and VF with method-specific relative L1 scaling.
11. Stores TGH/TWH library surfaces separately and uses their implicit Cartesian product in the VF value function.
12. Writes policy, pDCA, library, decomposition, saturation, settings, and freeze diagnostics.
13. Stage 1 exits before holdout materialization/evaluation.
14. Stage 2 verifies the frozen bundle, materializes the 22 holdout weeks, and evaluates exactly those accepted policies.
15. Stage 2 saves weekly results, site-level overtime/idle/turnover decompositions, and every final case-block assignment.

## Paper outputs

Stage 1 central outputs:

- `DATA_FREEZE.json`, `WEEK_SPLIT.csv`, `FROZEN_SETTINGS.json`, `SCIENTIFIC_SPEC.json`
- `FEATURE_MANIFEST.json`, `POLICIES.npz`, `REGULARIZATION.json`
- `ORACLE_TRAIN.csv`, `RA_PI_METRICS.json`
- `START_DECOMPOSITION.csv`, `VF_TRAJECTORY.csv`, `TRAIN_DECOMPOSITION.csv`
- `LIBRARY_SUMMARY.csv`, `SATURATION_SUMMARY.json`
- `TRAINING_FREEZE.json`, `RUN_STATUS.json`

Stage 2 central outputs:

- `ORACLE_HOLDOUT.csv`
- `FINAL_HOLDOUT_WEEKLY.csv`
- `FINAL_HOLDOUT_SUMMARY.csv`
- `FINAL_HOLDOUT_SITE_WEEKLY.csv`
- `FINAL_HOLDOUT_ASSIGNMENTS.csv`
- `TIE_RULE.json`, `EVALUATION_SETTINGS.json`, `FINAL_DECISION.json`

## Sensitivities / appendix notes

The primary result should remain fixed. Useful secondary checks can be reported separately where space permits:

1. `regular_template` versus `median_count_template` capacity;
2. behavioural response strength / distrust scenarios;
3. pooled two-site policy versus separately estimated site-specific policies as a negative-transfer diagnostic;
4. planner time/gap sensitivity where numerical brackets remain material;
5. turnover variation as an appendix robustness note if needed.

The primary turnover remains 30 minutes; a large turnover grid is not required for the main paper unless results prove unusually sensitive to it.
