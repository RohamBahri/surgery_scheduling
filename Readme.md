# Predictive & Stochastic Optimization for Elective Surgery Scheduling

This repository provides a Python prototype for experimenting with different
optimization strategies for elective surgery scheduling under uncertain case
durations.  It bundles data preparation utilities, predictive models, and
mixed-integer optimization models into a single reproducible workflow that can
be run from the command line.

## Approaches Implemented

The main script (`surgery_scheduling.py`) compares several ways of estimating
surgery durations and using those estimates inside an operating-room scheduling
model:

| Method | Key idea | Solver |
|--------|----------|--------|
| **Deterministic** | Uses booked time as fixed duration | Gurobi MIP |
| **SAA** | Two-stage stochastic program with empirical scenarios | Gurobi MIP |
| **LASSO-Predictive** | Duration predicted by one-off LASSO regression | Gurobi MIP |
| **KNN-Predictive** | \(k\)-NN predictor tuned on a cost proxy (overtime + idle) | Gurobi MIP |
| **Integrated** | Learn \(\theta\) via predict-then-optimize loss | Gurobi MIP |

A rolling-horizon simulation (default: **10 horizons × 7 days**) is executed,
and the code reports planned objective values, realized costs, overtime, idle
minutes, and acceptance/rejection statistics across horizons.

## Repository Layout

```
├── Readme.md                  ← project overview (this file)
├── requirements.txt           ← minimal Python dependencies
├── surgery_scheduling.py      ← end-to-end experiment runner
├── integrated/
│   └── run_integrated.py      ← standalone integrated-model experiment
└── src/
    ├── config.py              ← central configuration dataclasses
    ├── data_processing.py     ← ETL, feature engineering, rolling split
    ├── predictors.py          ← LASSO, KNN, XGBoost training utilities
    ├── solver_utils.py        ← deterministic, stochastic & predictive models
    ├── scheduling_utils.py    ← schedule extraction and KPI computation
    └── ...                    ← helpers for outputs and constants
```

## Prerequisites

* Python 3.9 or newer (the codebase uses dataclasses and type annotations).
* A working [Gurobi](https://www.gurobi.com/) installation with a valid license
  for `gurobipy`.
* Access to the historical elective surgery dataset referenced in
  `src/config.py` (defaults to `data/UHNOperating_RoomScheduling2011-2013.xlsx`).
  This dataset is not committed to the repository.

Optional but recommended:

* [XGBoost](https://xgboost.ai/) (`pip install xgboost`) to enable the gradient
  boosting predictor included in `predictors.py`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional: enable the XGBoost-based model
pip install xgboost
```

If Gurobi is not already licensed on your machine, follow
[Gurobi's quick-start guide](https://www.gurobi.com/documentation/) to obtain a
license and set the corresponding environment variables before running any
models.

## Preparing the Data

1. Place the Excel file containing elective surgery records in the `data/`
   directory (or update `CONFIG.data.excel_file_path` to point to its location).
2. Ensure the file has the columns referenced in
   `src/data_processing.py::load_data`—the loader normalizes column names to
   snake case and filters to non-emergency cases.
3. (Optional) Adjust warm-up weeks, minimum sample thresholds, or other filters
   in `src/config.py` to match your institution's data.

## Running the Rolling-Horizon Simulation

The quickest way to reproduce the experiment is to execute the top-level
script:

```bash
python surgery_scheduling.py
```

Key runtime options are controlled through the `CONFIG` object in
`src/config.py`:

* `CONFIG.debug_mode = True` — drastically shortens runtimes by shrinking the
  number of horizons, tightening time limits, and disabling the SAA model.
* `CONFIG.saa.scenarios` — controls the number of empirical duration scenarios
  used by the SAA model.
* `CONFIG.operating_room.*` and `CONFIG.costs.*` — govern block capacity,
  overtime caps, and penalty weights.

The script logs progress to stdout and writes results when `debug_mode` is
`False`:

* `outputs/results.json` — horizon-level decision summaries for each method.
* `outputs/agg_results.json` — aggregate KPIs across horizons.
* `outputs/final_theta.json` — parameters learned for the integrated model.

## Integrated Predict-then-Optimize Experiments

The `integrated/run_integrated.py` module contains a more fine-grained study of
learned duration adjustments (\(\theta\)) with optional Benders decomposition
and hyper-parameter sweeps. Run it directly for standalone experiments:

```bash
python -m integrated.run_integrated
```

Refer to `IntegratedConfig` in `src/config.py` for experiment settings such as
regularization strength, tolerance parameters, and the directory where multi-
parameter sweeps are saved.

## Extending the Project

* **Add new predictors:** create a training function in `src/predictors.py` and
  call it from `surgery_scheduling.py`. The scheduling models accept any
  predictor that produces point estimates or scenario samples.
* **Experiment with policies:** implement a new solver routine in
  `src/solver_utils.py` and register it in the main simulation loop to compare
  against existing policies.
* **Post-process results:** `src/output.py` centralizes JSON serialization and
  console summaries; extend it to add CSV export, visualization hooks, or
  database logging.

## Troubleshooting

* **`gurobipy` import errors:** confirm that Gurobi is installed, licensed, and
  that the Python bindings were added to your environment (`python -c "import gurobipy"`).
* **Missing data file:** update `CONFIG.data.excel_file_path` or create a
  symlink so the Excel loader can find the dataset.
* **Long runtimes:** enable debug mode or reduce `CONFIG.data.num_horizons`,
  SAA scenarios, and solver time limits.

## License & Citation

This codebase was originally created for research on elective surgery
scheduling policies.  Please cite the accompanying publication or institution
if you build upon it and review the license terms bundled with the repository.

