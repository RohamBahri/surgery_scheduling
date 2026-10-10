# W2b fixed-capacity weekly experiment (exploratory)

This branch forks `daily-experiment`; its existing files and results are untouched.
Only `w2b/`, `run_w2b.py`, one test, and packaging were added.

## Operational model

* Retain the **exact** cleaned 2011–2013 daily-experiment cohort.
* Each individual case can move between the weekdays its surgeon **actually
  operated in that week** and the surgeon's **top two weekdays learned from
  training history**. Only a destination on/after the recorded decision date
  is allowed, except the original date. Use `--max-move-days` to restrict moves,
  or `--max-moves` to cap the total number of cases changing weekdays per week.
* Preserve the **historically observed number of surgeon operating days**
  `k_s` in each week. A surgeon works in **one room per selected weekday**,
  with at least one case on each selected day. Cases can be repartitioned
  across these days; **historical surgeon-day lists are not kept intact**.
* Staffed room sessions are a **training-only weekday template**: the median
  number of historically active rooms for that group/weekday, choosing the most
  frequently used rooms. Every session incurs Idle cost even when empty.
  This is a **capacity proxy**, NOT actual staff rosters.
* Service-room compatibility is learned only on training dates. If the
  intersection of a surgeon's services and the staffed rooms is empty on a day,
  fall back to all staffed rooms and record the fallback.
* All cases are assigned; no deferral, hard overtime cap or cross-group
  coordination. Room load is sum of allocated durations plus 30 minutes between
  consecutive cases in occupied rooms. Cost is Idle + 1.75 Overtime across
  **all** staffed room sessions, plus optional cost of changing a case's weekday.
* A case's possible future availability and a surgeon's current-week working-day
  count are **explicit retrospective planning assumptions**, not observed
  advance availability. If considering the results as prospective, secure
  independent roster/patient-availability evidence first.

For a fixed assignment `z` with `R` occupied sessions out of `K`:
`J(z;d)=480K-sum(d)-30n+30R+2.75 OT(z;d)+move_penalty*moved(z)`.

## Three training objectives

Let `d(w)=booked+surgeon_response(Xw)`,
`V(d)=min_z J(z;d)`, `V(a)` its realized-duration oracle, and
`theta(e)=1.75*max(e,0)+max(-e,0)`.

| `--method` | Full-space analytical loss |
|---|---|
| `vf` | `V(d)+sum theta(a-d)-V(a)` |
| `gap` | `gamma*V(d)+max_z[J(z;a)-gamma*J(z;d)]-V(a)` |
| `spo` | `gamma*J(z_actual_oracle;d)+max_z[J(z;a)-gamma*J(z;d)]-V(a)` |

The last loss is **generalized SPO-style**, not SPO+ with its standard convexity
or consistency guarantees. `gamma >= 1`. Theoretical pointwise relation:
`realized regret <= gap <= vf` for `gamma>=1`, and `gap<=spo`
at the same gamma. All statements refer to **full** feasible schedule sets.

Learning uses a **restricted schedule library and local Powell search** (shared
Case-Error initialization), followed by exact follower and loss-augmented
Gurobi oracle calls to enrich the library. Neither the local optimizer nor the
library proxy is globally certified. The full adversarial oracle can be difficult
and may time out. Incomplete primary or adversarial solves are exposed in the
result JSON and never called optimal. No test period is used for training.

## Setup and commands

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/test_w2b.py -q

# Tiny initial pilot, 2 modest W2b training weeks, all 3 losses:
python run_w2b.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --group TWH-day --weeks 2 --max-cases 70 --method all \
  --scenario 0.8 30 --gamma 2 --outer 2 --inner-evals 150 \
  --seconds 120 --adversary-seconds 120 --threads 1

# Larger controlled sample; one loss:
python run_w2b.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --group TGH --weeks 8 --max-cases 150 --method gap \
  --scenario 0.5 60 --gamma 2 --outer 4 --inner-evals 500 \
  --seconds 600 --adversary-seconds 900 --threads 4

# Full cohort (potentially extremely expensive):
python run_w2b.py data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --group TGH --weeks 0 --max-cases 0 --method vf \
  --outer 6 --seconds 900 --threads 8
```

The default is deliberately small (`--group TWH-day --weeks 2
--max-cases 70`). `--weeks 0` and `--max-cases 0` lift those
limits. Run each of the three room groups and behavioral scenarios separately.
`--max-move-days 0` disallows case date moves but still uses a weekly model.
`--max-moves 12` allows at most 12 cases to change weekday in each week;
`--max-moves -1` (default) is unrestricted.
`--move-penalty` prices weekday changes, independently of duration.

For identical initialization to the previous daily experiment, pass
`--init-weights /path/to/models_TGH.json` (using the matching group/scenario).
Otherwise a response-aware smoothed casewise fit is estimated on the selected
training weeks. The latter **is not identical to the earlier Case-Error baseline**.

Each run writes `specification.json`, `initialization.json`, one JSON
trajectory per method, and `comparison.json` under `results/w2b/<digest>/`.
All complete Gurobi solves are cached. Rerun identical commands to reuse
those solves. Computational budgets (`--seconds`, etc.) affect feasibility
of completion; unfinished solves are NOT reused as optimal.

## Important constraints on inference

* These are **training-only**, post hoc research experiments. The daily test
  outcomes were already inspected; future evaluation on the same January–June
  2013 period cannot be described as untouched confirmatory validation.
* The methods optimize **local restricted-library proxies**, which are not
  proof of globally optimal weights.
* The moving-case assumptions may create nonclinical schedules without
  validated patient availability and staffing rosters. They must be justified
  before operational-effect claims.
* `--max-cases` can select unrepresentative small weeks: it is only a pilot
  sizing control, not a final paper cohort definition.
* No method establishes an outcome improvement until complete solves and
  independent realized-cost comparisons have been run and reviewed.
