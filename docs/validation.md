# Implementation validation

## Workbook audit

The source workbook has 60,567 rows. Its SHA-256 is:

\`\`\`
ce2830c04dba7b5a9444dea54ad6b56759e4cf260a902a85fbdbcfb351a4f308
\`\`\`

The cohort rules are unchanged by the October 2026 planner/feature refactor, so the expected audited cohort remains 24,506 training and 8,774 test cases across TGH, TWH main, and TWH day surgery. The audit now also exports cancellation reasons, booking extremes, repeat-patient summaries, specialty room/weekday concentration, extreme durations, and both room-eligibility fallback levels.

## Validation after the refactor

Run \`python -m pytest -q\`, then rerun \`audit\`, \`pilot\`, and \`checks\` on the supplied workbook before long training. The production planner is now one Gurobi room-pattern set-partitioning model; the previous compact/HiGHS timing results are not treated as validation of the new production path.

The automated tests cover:

- contamination scope, nested overlaps, and the strict 15-minute overlap boundary;
- leave-date-out training scores and full-training-only test scores;
- pairwise spread hand calculations and mean-effect shrinkage;
- weekday service-room eligibility plus both fallback levels;
- the production Gurobi pattern planner and compact validation formulation against exhaustive small assignments;
- deterministic cost / largest-load / alphabetical ties;
- response floors, behavioral mapping, and response-limited oracle bounds;
- resumable Case-Error/VF updates and exact-plan requirements;
- penalty reproducibility and zero-policy pull diagnostics;
- frozen test-policy fingerprints;
- case-level output uniqueness and room-day load reconciliation;
- paired calendar-week bootstrap reporting and common response-oracle brackets.

The exact numerical package versions are pinned in \`requirements-lock.txt\` and the active Python/Gurobi/NumPy/pandas/SciPy/openpyxl versions are included in each run identity. A source or environment change therefore starts a distinct run.

## Local checks before training

1. \`--stage audit\`: confirm the retained counts, booking grid, specialty/eligibility diagnostics, and nonempty daily eligibility.
2. \`--stage pilot\`: confirm P1 daily plans are exact and P2 oracle brackets are valid.
3. \`--stage checks\`: solve every training Booked and realized-duration plan with the single Gurobi pattern planner and inspect Historical versus Booked room use and load distributions.
4. Only after those checks are satisfactory should the source be frozen for \`baselines\`, \`pilot-vf\`, \`train\`, and the single \`evaluate\` run.
