# UHN daily planning experiment

This repository implements one experiment on the UHN July 2011–June 2013 workbook. It uses TGH, TWH main, and TWH day surgery. Cases keep their dates and room groups; rooms may be released. PMH is excluded.

The five methods are Booked, Shift, Case-Error, VF-Direct, and VF. The four response scenarios are `(alpha, h) = (0.5, 30), (0.5, 60), (0.8, 30), (0.8, 60)`. VF-Direct is fitted once per group and evaluated through each response.

## Run

Python 3.10+ and a full Gurobi license are required for the complete experiment. Small tests run with Gurobi's restricted license.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python run_experiment.py /path/to/UHNOperating_RoomScheduling2011-2013.xlsx --stage audit
python run_experiment.py /path/to/UHNOperating_RoomScheduling2011-2013.xlsx
```

The default `all` stage performs the audit, solver preflight, P1/P2, training checks, penalty calibration and baselines, P3, VF-Direct/VF, and one frozen test evaluation. It never selects settings from test results. Results and private row manifests stay under `results/`, which is excluded from Git.

Use `--stage pilot`, `checks`, `training-oracles`, `baselines`, `pilot-vf`, `train`, or `evaluate` to stop at an intermediate stage. Prerequisites are reused. `train` includes P3 and does not evaluate the test set. `results/current_run.json` identifies the output directory.

## Resume

Run the same command again. Completed exact daily solves, accepted policy updates, baseline trials, growing schedule libraries, oracle cuts, and feasible oracle bounds are checkpointed in SQLite. Each day has its own database under `days/`, with one writer; fitted models use the run-level database. Runtime budgets do not change the run identity.

```bash
python run_experiment.py /path/to/workbook.xlsx --stage train --seconds 900 --workers 4 --threads 1
python run_experiment.py /path/to/workbook.xlsx --stage evaluate --oracle-seconds 120 --refine-oracles
```

`--seconds` is the budget for a complete daily planner call, including its tie stages, or one convex policy update. An unfinished planner remains pending and is retried; it is never treated as exact or dropped from the reporting sample. Exit code 2 means the requested stage is pending; see `status.json`. Interruptions retain completed work. An unfinished response oracle is a valid bracket and does not prevent reporting if its bounds are consistent. `--refine-oracles` spends an additional budget on those brackets without refitting policies.

The fixed planner rule is minimum cost, then minimum largest room load, then the alphabetical surgeon-to-room assignment. Compact assignment is tried first; an unfinished solve uses the remaining budget for a complete room-pattern formulation with HiGHS. Its admissible patterns follow a feasible cost bound and a dual LP screen, with no limit on surgeons per room; interchangeable rooms share capacity variables. Only a proven solution of all required stages is exact. VF library updates and realized-duration benchmarks require primary optimality only.

Independent days run in separate processes (`--workers`, default up to 4); `--threads` is the solver thread count per worker. Each Shift day runs its shifts sequentially with warm starts. A failed worker batch resumes saved day tasks with fewer workers; each day has a separate checkpoint database, so workers never write the same database concurrently. Full training response-oracle sweeps are optional (`--stage training-oracles`); the required path uses P2's representative training days and benchmarks every test day. An unfinished oracle retains its bracket.

Case-Error fits run in blocks of 30 updates. A capped fit is saved with explicit convergence/cap flags; rerunning the stage continues it. Test evaluation requires converged Case-Error fits. P3's first VF iteration is reused by the main fit. Source changes create a new run identity; freeze the implementation before a long run.

## Outputs

- `audit/`: cohort flow, retained/excluded Excel row IDs, daily cleaning losses, assignability, chronological encoder metadata and features.
- `pilots_P1_P2.json`, `pilot_P3.json`: training-only timing and certification records.
- `checks/`: historical schedules, the projection minimizing moved cases under the one-room rule, Booked, and oracle costs/bounds.
- `models_*.json`, `training_diagnostics.json`, `training_certificates.csv`: coefficients, Shift searches, fitting status, bound activity, VF trajectories, final libraries, and learned/zero-policy bounds on those same libraries.
- `frozen_test_policies.json`: the weights and shifts fixed before test evaluation.
- `report/`: daily results, method summaries, oracle brackets, opportunity ratios, paired weekly confidence intervals, and the shared bootstrap indices.
- `checkpoints.sqlite`, `days/*/checkpoints.sqlite`, `solver_log.jsonl`: durable task state and every optimizer call, including stage, status, objective, bound, time, dimensions, and solver version.

The implemented cohort contains 24,506 training cases and 8,774 test cases in the supplied workbook. These are audited outputs, not count constraints. The exact inclusion rules, date interpretation, and input/retained-row hashes are recorded. The study concerns retained elective workloads, not complete historical hospital workloads.

## Code

| Module | Responsibility |
|---|---|
| `data.py` | Cohort, earlier-date scores, and daily instances |
| `planner.py` | Response, assignment, exact ties, and response-limited oracle |
| `learning.py` | Penalty, Shift, Case-Error, and VF updates |
| `experiment.py` | Ordered stages, checkpoints, and frozen evaluation |
| `reporting.py` | Matched summaries and weekly inference |
| `storage.py` | Atomic files and SQLite state/logs |

See [validation results](docs/validation.md), [the experiment specification](docs/experiment.md) and [the mathematical implementation notes](docs/theory.md).
