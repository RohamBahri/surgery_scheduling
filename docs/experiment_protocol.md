# Multi-scenario experiment protocol

`experiment_registry.json` is the committed source of truth for behavioral scenarios and planned comparisons. Do not add, remove, substitute, or select scenarios after holdout results are observed.

## Frozen behavioral design

The registered training/evaluation scenarios are:

| name | alpha | h (min) | role |
| --- | ---: | ---: | --- |
| `primary` | 0.8 | 30 | primary |
| `lower_responsiveness` | 0.5 | 30 | secondary |
| `narrower_tolerance` | 0.8 | 15 | secondary |
| `broader_tolerance` | 0.8 | 60 | secondary |

Rows of the response matrix are **training scenarios**. Columns are **response conditions used at evaluation**. Diagonal cells are correctly specified; off-diagonal cells measure behavioral misspecification. Every matrix solve uses the primary deterministic evaluation protocol: WorkLimit 1200 per site, 0.05% native-Psi gap target, one thread, seed 42, fixed ordering, and the reviewed emergency-wall retry rule.

The projected hindsight benchmark uses the reachable correction interval based on `min(h, display_cap)` and the case-specific one-minute recommendation floor. It is a benchmark, not a scheduling-performance ceiling or lower bound.

## Reusable training backbone

Before scenario training, solve/refine the behavior-independent training Oracle and BOOKED plans once:

```bash
python run_final_paper.py shared-plans \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/shared_plans_v1 \
  --cores 15 \
  --kind both \
  --oracle-seconds 1800 \
  --booked-seconds 300 \
  --booked-gap 0.01
```

If difficult weeks need additional effort, create a new immutable refinement artifact using the old assignments as MIP starts:

```bash
python run_final_paper.py shared-plans \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/shared_plans_v2 \
  --warm-start-root artifacts/shared_plans_v1 \
  --cores 15 \
  --kind both \
  --oracle-seconds 3600 \
  --booked-seconds 900 \
  --booked-gap 0.002
```

Choose one refined shared artifact and use that exact artifact for every registered training scenario. Its manifest fingerprints ordered case IDs and processed durations separately for Oracle and BOOKED, weekly structure/eligibility, costs/turnover, relevant mathematical-model source files, solver version/parameters, registry hash, input workbook, plan files, Git commit, and refinement parent. Cached lower bounds are rejected if any of these provenance checks fail.

Every weekly Gurobi solve writes a persistent file log under the run's `gurobi_logs/` directory even when console output is quiet.

## Training

Use the same clean Git commit and shared-plan artifact for every registered scenario. For example:

```bash
python run_final_paper.py train \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --shared-plans-root artifacts/shared_plans_v2 \
  --scenario-name primary \
  --artifact-root artifacts/train_primary \
  --cores 15 \
  --max-wall-minutes 720
```

Repeat for `lower_responsiveness`, `narrower_tolerance`, and `broader_tolerance`. `--alpha` and `--h` may be supplied for readability, but they must exactly match the committed registry entry. Unregistered scenarios are rejected.

All registered scenarios use the same declared Stage-1 wall budget/stopping rules. Different VF iteration counts or termination reasons do not automatically invalidate comparisons; they are recorded as budget-limited algorithm outcomes and are not convergence/optimality certificates.

## Evaluation policy

There is no pre-holdout seal and no single-consumption rule. Completed training
bundles may be evaluated repeatedly, including after inspecting earlier holdout
results. Scientific provenance and training-bundle compatibility checks remain
in force, but they do not prevent additional holdout analysis.

## Holdout evaluation

Use the pooled matrix runner for the paper evaluation:

```bash
python run_final_paper.py matrix \
  --experiment-root artifacts/experiment_v1 \
  --training-root artifacts/train_primary \
  --training-root artifacts/train_lower_responsiveness \
  --training-root artifacts/train_broader_tolerance \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15
```

The matrix engine solves the realized oracle and BOOKED map once, deduplicates exactly identical duration maps, checkpoints every completed week/map job, and replays those checkpoints through the reviewed Stage-2 reporting code. The row-by-row `evaluate` command remains available for individual repeatable evaluations, but it is not the efficient engine for the paper response matrix.

If a matrix evaluation is interrupted, rerun the same matrix command with `--resume`. Compatible completed week/map checkpoints are reused even across provenance-only commits when the solver-relevant mathematical source is unchanged. A fresh evaluation can also be started intentionally because holdout access is not a one-shot lock.

The matrix runner writes combined `RESPONSE_MATRIX_WEEKLY.csv` and `RESPONSE_MATRIX_SUMMARY.csv` under both the experiment root and `holdout_matrix/`, using one experiment-wide oracle bracket for cross-row comparisons.

## Numerical accounting

The exact reconstructed assignment is independently reevaluated and that feasible Phi is always the reported incumbent. Solver-summed Phi is only an audit value. The accounting allowance is

```text
min(0.5, max(0.1, 2e-6 * scale))
```

where `scale=max(1, |recomputed Phi|, |solver-summed Phi|)`. Under the current numeric policy, finite solver-versus-reconstruction differences are diagnostic warnings rather than live kill switches; the independently recomputed feasible Phi is authoritative. Certificate validity is protected separately by rejecting a solver lower bound that exceeds the feasible objective beyond numerical tolerance.
