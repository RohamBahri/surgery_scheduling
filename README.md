# Surgeon-aware operating-room scheduling study

One retrospective research experiment on the UHN 2011–2013 operating-room
workbook. The project learns case-duration recommendations that pass through
a **specified** surgeon response before a weekly mixed-integer planner.

**This is not a clinical deployment tool.** The historical data contain no
responses to displayed advice. Every behavioral result is conditional on
assumptions, and January–June 2013 was inspected during development (an
exploratory retrospective evaluation, not an untouched holdout).

## Installation

Python 3.12 and a working Gurobi license:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Keep the private workbook under `data/`; all data and outputs under
`data/` and `artifacts/` are ignored by Git.

## Reproduce the paper experiment

Run these commands at the repository root. The training and scheduling steps
are substantial; they run locally on the research machine, not in GitHub CI.

**1. Prepare the single dataset**

```bash
python -m pilot.prepare_data \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/context.pkl
```

This creates the case features, surgeon-eligible scheduling components and
auditable cohort/quality tables. It uses July 2011–December 2012 for training
and January–June 2013 for evaluation.

**2. Tune the L1 penalties on earlier training data**

```bash
python -m pilot.tune \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/lambda_tuning --cores 15
```

This creates an independent July 2011–June 2012 fitting and July–December
2012 validation split, using eight evenly spaced validation weeks for economical\nweekly cost comparisons. It screens λ = 0.03, 0.1, 0.3 separately for
Response-aware and Decision-aware, under two anchor response assumptions.
Decision-aware uses shortened fits **only for screening**. The chosen λ for
each method is shared across all final scenarios. No 2013 outcomes are used
for tuning. The selected file is `artifacts/lambda_tuning/selected_hyperparameters.json`.

**3. Train and evaluate the eight prespecified response scenarios**

```bash
python -u -m pilot.experiment --stage all \
  --context artifacts/context.pkl \
  --tuning artifacts/lambda_tuning/selected_hyperparameters.json \
  --out artifacts/paper_experiment --cores 15 --plan-seconds 300
```

To split a run across sessions, use `--stage train` followed by
`--stage evaluate` with *exactly the same arguments and output directory*.
Finished policy and scheduling checkpoints are reused only when their
scientific settings match. Use a new output directory after changing settings.

Default assumptions:

- Saturating surgeon response; adoption α = 0.5 or 0.8 with credibility
  thresholds h = 20, 30 or 60 minutes, plus (0.3, 15) and (0.8, 90).
- Weekly physical room-days with fixed 480-minute capacity, no turnover,
  eligibility restrictions and relaxed observed surgeon-day retention.
- Primary scheduling cost: 1 × idle minutes + 1.75 × overtime minutes.
- A deterministic hash chooses among optimal primary schedules.
- Shared 83 features (77 case features + 6 structural planning features) for
  all three learned comparators.
- A displayed-duration accuracy safeguard of 5% relative to the *fitted*,
  non-globally-certified Accuracy-trained reference.
- Scenario-specific accuracy-penalty μ selected using training data only.
  The default tries 2, then 4 and 8 if necessary; alternative candidate
  lists can be supplied explicitly.

The comparison consists of exactly six policies:

| Paper name | Description |
|---|---|
| Booked | Planner uses the surgeon's historical duration |
| Accuracy-trained | Fitted displayed-duration MAE policy; ignores planning |
| Response-aware | Asymmetric case-local post-review loss; no scheduling in training |
| Decision-aware | Proposed direct value-function MM/pDCA policy |
| Perfect advice | Hindsight accuracy-optimal displayed advice subject to surgeon review |
| Hindsight | Planner uses realized durations directly |

Perfect advice is an **informative hindsight benchmark**, not a guaranteed
scheduling-cost lower bound. Hindsight is the unrestricted cost reference.
An expensive bilevel "best under review" method is intentionally excluded.

Results are aggregated over **complete calendar weeks certified for every
method and scenario**. Missing components are never interpreted as complete
weekly totals. The experiment saves component- and week-level results and
paired bootstrap intervals. Prediction MAE is in minutes per case; scheduling
cost, overtime and idle time are reported per calendar week.

**4. Test response misspecification without retraining**

```bash
python -u -m pilot.experiment --stage cross-evaluate \
  --context artifacts/context.pkl \
  --tuning artifacts/lambda_tuning/selected_hyperparameters.json \
  --out artifacts/paper_experiment --cores 15 --plan-seconds 300
```

Two representative trained scenarios are evaluated with lower actual
acceptance and with a nonmonotone true response with a lower threshold.
**Displayed recommendations stay fixed** when the true response changes.

**5. Generate tables, figures and the source-of-truth numbers**

```bash
python -m analysis.report \
  --run artifacts/paper_experiment --context artifacts/context.pkl
```

This consumes frozen results without calling an optimizer. Outputs under
`artifacts/paper_experiment/publication/` include:

- `numbers.json`: verified numerical source for paper revisions;
- `tables/*.csv`: main results, weekly comparisons, site breakdowns,
  accuracy statistics and paired uncertainty;
- `figures/*.svg` and `*.png`: accuracy versus weekly cost,
  savings by scenario, overtime/idle, weekly differences and booking errors;
- `summary.md`: concise overview.

Not every generated table or figure must be included in the manuscript.

## Repository layout

| Path | Purpose |
|---|---|
| `surgery/data.py` | Audited cohort and leakage-conscious history features |
| `surgery/weekly_data.py` | Eligible room-day weeks and independent components |
| `surgery/weekly_planner.py` | Scheduling model, load, and cost identity |
| `surgery/schedule_solver.py` | Exact primary solver, deterministic tie-breaking, certificates |
| `surgery/learning.py` | Surgeon response, DC objectives, pDCA, value-function majorization |
| `surgery/features.py` | Six structural planning-context features |
| `pilot/prepare_data.py` | Single-workbook study context |
| `pilot/tune.py` | Training-only L1 screening |
| `pilot/experiment.py` | Main six-policy, eight-scenario run |
| `pilot/policy_runner.py` | Reusable single-policy training invocation |
| `pilot/policy_training.py` | Direct decision-aware training and planner cache |
| `pilot/response_stress.py` | Fixed-display behavioral stress tests |
| `pilot/operational_audit.py` | Difficult room-load diagnostics |
| `analysis/report.py` | CSV/JSON and SVG/PNG publication outputs |

The code intentionally does not support unrelated datasets, large-scale
services or a zoo of alternative learning algorithms.
