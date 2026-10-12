# Weekly supported-day planning (experimental)

This branch starts from `daily-experiment`. It reuses `surgery.data.build_cohort`
unchanged, including the 24,506/8,774 train/test split. The old daily experiment
remains available. `weekly_data.py` defines eligibility; `weekly_planner.py`
defines optimization; `run_weekly.py` is a **training-only** audit/benchmark.

## Decisions represented

- One group × one Monday–Friday week; physical rooms active in the retained
  historical group-week are fixed, paid room-day sessions (480 min).
- Original surgeon-day/room relationships are preserved as eligible sessions.
  Cases may individually move among a surgeon's supported sessions; all cases
  must be assigned. Multiple surgeons per room and multiple rooms per surgeon
  are permitted, but no new unrestricted same-day room access is added.
- A surgeon observed on **one day** with **at least two retained cases** can
  optionally use **one more weekday**. Evidence: a valid performed surgery on
  that weekday *before* the planning week (one distinct date suffices), or
  same-week performed activity excluded during cleaning. The eligible physical
  room must already be active in the target week and historically used by that
  surgeon (or be the matching excluded-activity room). The best supported
  (weekday,room) candidate is selected deterministically.
- Each originally observed surgeon operating day keeps at least one case;
  the extra day is optional. No entire-list relocation, postponed case,
  endogenous staffing, or within-day sequencing.
- Cost = Idle + 1.75 Overtime over **every staffed room-day**, including
  unused sessions. Load = case room durations + 30 minutes between cases.
- Planner: minimum predicted cost, then minimum changed room-day assignments,
  then exact lexicographic ordering of case assignments (sorted case IDs).
  The canonical stage is solved in bounded mixed-radix chunks to avoid very
  large coefficients. Gurobi uses one thread and seed 0. No realized durations
  enter the planner or tie-breaking. The benchmark records primary and
  tie-stage runtimes separately; extra tie stages are not assumed free.
- Report both room overload and surgeon-day case-minute overload. The latter
  is a diagnostic only; no surgeon sequencing/overlap guarantee is asserted.

Weekdays/rosters and same-week excluded activity are **retrospective proxies**.
We cannot infer patient/surgeon availability from the workbook with certainty.
Historical prior evidence is filtered to dates strictly before each target week.
Do not describe this as a prospective availability reconstruction.

## Run

```bash
git switch weekly-supported-days
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest tests/test_weekly.py -q

# Structural audit: no Gurobi optimization, initial 2 group-weeks per group
python run_weekly.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage audit --weeks 2 --output artifacts/weekly_audit_2

# Paired Booked/hindsight and gap-VF adversarial oracle timings
python run_weekly.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage benchmark --weeks 2 --adversary --seconds 60 \
  --adversary-seconds 60 --gamma 2 --output artifacts/weekly_pilot_2

# Full training structural audit, then exact solver benchmark
python run_weekly.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage audit --weeks 0 --output artifacts/weekly_audit_full
python run_weekly.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage benchmark --weeks 0 --adversary \
  --output artifacts/weekly_benchmark_full
```

Outputs: `manifest.json`, `components.csv`, and (for benchmark)
`benchmark.csv`, `summary.json`. Never overwrite an existing artifact
directory. Unfinished or tie-incomplete solves are flagged and cause nonzero
exit, rather than being mislabeled optimal.

Use `--prior-dates 2` as a robustness check. `--group TGH` or another
room group and `--weeks N` restrict *group-weeks* for fast pilots.

## Model/method boundary

The optimizer exposes `solve_week(component, reviewed_durations)`,
`solve_adversary(component, reviewed_durations, gamma)`, and
`evaluate(component, assignment, actual_durations)`. Each component
decomposes by the *full eligible* room-day graph, including second-day edges.
The loss-augmented maximization enforces exact realized overtime. Gurobi
returns valid bounds; unfinished optimizations do not become certificates.

This branch benchmarks planner/oracle tractability before selecting a policy
optimizer. It **does not** train a new VF, gap-VF, or SPO policy and should not
be interpreted as demonstrating any learned-policy gains. Keep January–June
2013 out of model selection.


## CPU parallelism and targeted retry

Use **15 total CPU cores**, not 15 Gurobi threads on every small problem.
For full-cohort experiments run 15 independent spawn-based processes, with
one Gurobi thread per process (`--workers 15 --threads 1`). Runtime output
and CSV order are deterministic by component index; model identities, costs,
and eligibility are unchanged. A worker raises on any numerical inconsistency.
Solver wall-clock times and optimization trajectories may differ with CPU load.

```bash
# Future full-cohort benchmarks: 15 concurrent single-threaded oracles.
caffeinate -i python3.14 run_weekly.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage benchmark --weeks 0 --adversary --gamma 2 \
  --workers 15 --threads 1 --seconds 60 --adversary-seconds 60 \
  --output artifacts/weekly_parallel_new

# Resolve only previously uncertified components. One hard MILP can use
# all 15 threads internally instead of spinning up 15 mostly idle workers.
caffeinate -i python3.14 run_weekly.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage benchmark --weeks 0 --adversary --gamma 2 \
  --retry-from artifacts/weekly_benchmark_full/benchmark.csv \
  --workers 1 --threads 15 --seconds 120 --adversary-seconds 600 \
  --output artifacts/weekly_retry_600
```

Retry requires the original `manifest.json` beside the input CSV, and
verifies the dataset fingerprint, cohort/component structure, prior-date
eligibility and gamma before reusing certified rows. Save to a NEW directory.
Use this only when the mathematical planner/eligibility sources have not
changed. Numerical times in retained rows are from the original run, not a
fresh 15-core benchmark. No existing artifact is silently overwritten.


## Workload validation before changing the planning objective

The core benchmark is unchanged. To investigate the room-days above 720
minutes, rerun only components flagged in the certified full-cohort benchmark,
using exact Gurobi solves. Compare the existing tie (primary cost, fewest
moves, canonical) against **primary cost, minimum peak PREDICTED room load,
fewest moves, canonical**.

```bash
caffeinate -i python3.14 run_weekly_diagnostics.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --benchmark artifacts/weekly_retry_600/benchmark.csv \
  --workers 15 --threads 1 --seconds 120 \
  --output artifacts/weekly_load_audit
```

The `--benchmark` directory must contain the corresponding `manifest.json`.
The script reconstructs the cohort and checks the workbook fingerprint and
baseline realized costs component by component. Output: `components.csv`,
`rooms.csv`, and `summary.json`. The audit exposes a conservative
room-specific forced-case load lower bound; it is **not** a full infeasibility
proof for less constrained cases. Judge balancing by both realized overload
and realized cost. Balancing is *not* made the default planner rule.

## Learning objective pilot (TRAINING PERIOD ONLY)

Evaluate a common, intentionally small recommendation class
`u_i = offset + slope * (booked_i-180)/180`, with the same response map,
grid, training weeks, validation weeks, and exact weekly planner for all methods.
The methods differ only in their **training objective**: Shift (constant
advice, case MAE), Case (affine advice, case MAE), original VF, gamma-gap-VF,
and an SPO-style upper bound. BOOKED is the zero-advice reference. Optimize
each objective by exhaustive finite-grid selection (not pDCA or end-to-end
gradient learning) and compare realized regret on strictly *later training
weeks*. Test data from January–June 2013 are excluded.

```bash
caffeinate -i python3.14 run_weekly_learning.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --start-week-index 20 --train-weeks 3 --validation-weeks 2 \
  --alpha .8 --h 30 --gamma 2 \
  --workers 15 --threads 1 --seconds 60 \
  --output artifacts/weekly_learning_pilot
```

Files: `grid_results.csv` (all candidate/period results),
`policies.csv` (one training-selected policy per method/group), and
`validation.csv` (matched training and later-week regret/cost summaries).
Use `--resume` with the identical command/output if interrupted.
Uncertified solves are never used to select a policy. More demanding runs
can increase `--train-weeks`, `--validation-weeks`, and grid density
in a **fresh** output directory. Choose those settings without consulting
2013 test results. This pilot is *diagnostic*: success would justify
implementing a richer case-feature policy optimizer, not establish
publishable out-of-sample gains.
