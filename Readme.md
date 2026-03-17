# Incentive-Aware Operating Room Scheduling

Modular experiment platform for comparing elective surgery scheduling methods
under uncertain durations.  Designed around a shared weekly evaluation engine
with pluggable method classes.

## Project Structure

```
├── run_experiment.py           ← entry point
├── pyproject.toml              ← dependencies and metadata
└── src/
    ├── core/
    │   ├── config.py           ← all tunable parameters
    │   └── types.py            ← canonical data types (CaseRecord, WeeklyInstance, …)
    ├── data/
    │   ├── loader.py           ← Excel loading, cleaning, rare-category recoding
    │   ├── splits.py           ← warm-up / pool temporal split
    │   └── capacity.py         ← OR block capacity from historical room counts
    ├── planning/
    │   ├── instance.py         ← builds WeeklyInstance objects from the pool
    │   ├── schedule.py         ← extracts ScheduleResult from Gurobi models
    │   └── evaluation.py       ← shared actual-cost evaluator (all methods use this)
    ├── solvers/
    │   └── deterministic.py    ← single-stage assignment MIP (paper Section 4)
    ├── methods/
    │   ├── base.py             ← abstract Method class (fit / plan interface)
    │   ├── registry.py         ← method collection for the experiment runner
    │   ├── booked.py           ← baseline: raw booked durations
    │   └── oracle.py           ← lower bound: actual durations (clairvoyant)
    └── experiments/
        └── runner.py           ← rolling-horizon evaluation loop
```

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **Gurobi 10+** with a valid license (`python -c "import gurobipy"` must work)
- The UHN dataset Excel file (not committed to the repository)

### 2. Create a virtual environment and install dependencies

```bash
cd or_scheduling
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -e .
```

This installs the project in editable mode along with all dependencies listed
in `pyproject.toml` (pandas, numpy, scikit-learn, gurobipy, openpyxl).

### 3. Place the dataset

Copy the Excel file into a `data/` directory at the project root:

```
or_scheduling/
  data/
    UHNOperating_RoomScheduling2011-2013.xlsx
```

Or pass a custom path at runtime with `--data`.

### 4. Run the experiment

```bash
# Full run (53 out-of-sample weeks, ~10 min depending on hardware)
python run_experiment.py

# Quick test (3 weeks, fast solver limits)
python run_experiment.py --quick

# Custom data path and horizon count
python run_experiment.py --data path/to/data.xlsx --horizons 10

# Show Gurobi solver output
python run_experiment.py --quick --verbose
```

Results are written to `outputs/`:
- `horizon_results.csv` — one row per (method, week) with all KPIs
- `aggregate_summary.json` — mean and median KPIs per method

## Adding a New Method

1. Create `src/methods/your_method.py` implementing `Method`:

```python
from src.methods.base import Method

class YourMethod(Method):
    def __init__(self, config):
        super().__init__(name="YourMethod", config=config)

    def fit(self, df_train):
        # learn from warm-up data
        ...

    def plan(self, instance):
        # produce a ScheduleResult for one week
        ...
```

2. Register it in `run_experiment.py`:

```python
registry.register(YourMethod(CONFIG))
```

That's it.  The runner handles everything else: instance construction,
evaluation, result collection, and output.

## Configuration

All parameters live in `src/core/config.py`.  Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.warmup_weeks` | 52 | Training period length |
| `data.horizon_days` | 7 | Planning horizon (1 week) |
| `data.num_horizons` | 53 | Number of test weeks |
| `capacity.block_capacity_minutes` | 480 | OR block length (8 hours) |
| `capacity.reduction_fraction` | 0.5 | Capacity reduction from historical |
| `costs.overtime_per_minute` | 15.0 | Overtime penalty ($/min) |
| `costs.idle_per_minute` | 10.0 | Idle penalty ($/min) |
| `costs.rejection_per_booked_minute` | 20.0 | Rejection penalty ($/booked min) |

## Design Principles

- **Method-centric, not predictor-centric.**  Every method implements
  `fit()` + `plan()`.  The paper's behavioral CCG method is a first-class
  structural planner, not a duration predictor forced through a generic solver.

- **One evaluation path.**  Every method's schedule is evaluated by the same
  `evaluate()` function using realized durations.  No method-specific KPI logic.

- **Typed data contract.**  `CaseRecord`, `WeeklyInstance`, `ScheduleResult`,
  and `KPIResult` are the lingua franca.  No raw-dictionary passing between
  modules.
