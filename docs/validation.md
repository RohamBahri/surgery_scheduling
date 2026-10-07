# Implementation validation

## Workbook audit

The source workbook has 60,567 rows. Its SHA-256 is:

```
ce2830c04dba7b5a9444dea54ad6b56759e4cf260a902a85fbdbcfb351a4f308
```

The implemented rules produce 24,506 training and 8,774 test cases across the three included groups. The sorted retained-row list, encoded with the repository's `digest` function, has SHA-256:

```
69a594a89b5f87ac99af8b6d11460b5cd62099a84f8f5e4475904154a0b0a45e
```

The raw booking grid is verified. The training median morning-list count is 25, yielding 26 closure dates. There are 124 invalid records without a usable date in the included room registry. Every retained group-day is assignable and no surgeon-day uses the eligibility fallback. The private row manifests and features are generated locally, not committed.

## Numerical and recovery checks

The test suite covers:

- Contamination restricted to otherwise eligible records and keyed by room group.
- Nested room overlaps and the strict 15-minute boundary.
- No same-date or future-date outcome use in history scores; frozen test histories.
- Unequal-size ANOVA shrinkage and its nonpositive-variance case.
- Both planner tie rules against exhaustive assignment enumeration.
- The room-opening cost identity and outcome-blind deterministic ties.
- Reachable response intervals, the duration floor, and the DC decomposition.
- Response-limited oracle values in hand-checkable reachable/unreachable examples.
- A master without a proven optimum remaining inexact despite a successful follower check.
- Retrying incomplete cached plans and invalidating changed eligibility.
- Convex updates, training-bound descent, and comparison with the zero policy on identical libraries.
- Oracle-ratio brackets, shared calendar-week bootstrap indices, comparator percentages, and incomplete-report suppression.
- The full three-group/four-scenario pipeline on synthetic cases, including exact resume without additional optimizer calls and rejection of changed test weights.

Run `python -m pytest -q` for the current results.

## Runtime limits

The available Gurobi 13.0.3 license is size limited. The full-run preflight detects its inability to solve a representative learning-size model immediately after the audit and writes a pending status; no learning is launched.

Training-only daily pilots were performed with short budgets. Small instances, including the sampled day-surgery instances, completed. Some larger cases exhausted the budget or exceeded the quadratic-model license limit. A longer comparison on TWH-main 2012-10-18 completed Booked planning in approximately 4.8 seconds with squared loads and 10.0 seconds with lexicographic loads. TGH 2011-10-03 remained unfinished under both tie rules after 30 seconds. These timings are observations on individual instances, not a runtime claim for the complete experiment. They do not justify selecting the alternative balancing rule globally.

Full-size P1/P2 completion, P3, coefficient-bound checks on learned UHN policies, and the complete training/test run still require a full solver license and adequate solve budgets. No empirical policy-performance results are claimed by this validation record.
