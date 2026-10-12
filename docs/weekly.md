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
  then a fixed outcome-blind index (Gurobi single-thread/seed 0). The tertiary
  index is deterministic but is **not a mathematically unique lexicographic
  ordering of each case**: different assignments can share its score.
  No realized durations enter the planner or tie-breaking.
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
