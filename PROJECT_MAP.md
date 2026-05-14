# Project Map

This file is a human map of the repository. The main reusable code lives in
`src/`. The `scripts/` files are reporting or one-off analysis scripts. The
`tests/` folder is useful as executable examples, but it is not required to
understand the research pipeline.

## Main Flow

1. `scripts.run_experiment` builds a `Config`, registers methods, and calls
   `src.experiments.runner.run_experiment`.
2. `src.experiments.runner` loads and splits the data, builds weekly planning
   instances, fits each method, plans each week, evaluates schedules, and writes
   artifacts.
3. `Booked` and `Oracle` methods call the deterministic weekly solver directly.
4. `VFCG` first fits estimation models, trains weights with the VFCG stack, and
   then plans each week with post-review durations.

## Root Files

| Path | What it does |
| --- | --- |
| `.github/workflows/estimation-tests.yml` | GitHub Actions workflow that installs the package and runs estimation tests. |
| `.gitignore` | Keeps generated artifacts, data, caches, virtualenvs, and tmp files out of git. |
| `.DS_Store` | macOS Finder metadata; it has no project logic. |
| `README.md` | Short setup, CLI, and artifact-location overview. |
| `PROJECT_MAP.md` | This file. |
| `pyproject.toml` | Package metadata, dependencies, editable-install setup, and CLI entry points. |
| `requirements.txt` | Flat dependency list for environments that do not use `pyproject.toml`. |
| `run_experiment.py` | Small root-level wrapper for `scripts.run_experiment:main`. |
| `gurobi.env` | Local Gurobi solver configuration, if present. |

## `src/`

| Path | What it does |
| --- | --- |
| `src/__init__.py` | Describes the source-package layout. |
| `src/validation.py` | Logs sanity warnings for a completed weekly schedule. |

## `src/core/`

| Path | What it does |
| --- | --- |
| `src/core/__init__.py` | Public entry point for shared config, types, paths, and schedule columns. |
| `src/core/config.py` | All dataclass configuration objects, including data, costs, solver, estimation, and VFCG settings. |
| `src/core/types.py` | Shared names and containers: column constants, domain constants, cases, blocks, weekly instances, schedules, and KPI rows. |
| `src/core/column.py` | Converts solved schedules into reusable schedule columns and computes column costs. |
| `src/core/paths.py` | Creates run-specific artifact directories and paths. |

## `src/data/`

| Path | What it does |
| --- | --- |
| `src/data/__init__.py` | Public entry point for data loading, splitting, scoping, capacity, and eligibility helpers. |
| `src/data/loader.py` | Reads the UHN workbook, cleans cancelled/emergency/bad-timestamp cases, standardizes IDs, and adds time features. |
| `src/data/splits.py` | Splits cleaned data into a warm-up training set and rolling planning pool. |
| `src/data/scope.py` | Filters data to configured planning weekdays and sites. |
| `src/data/capacity.py` | Learns regular room/site weekday pools and builds candidate block calendars. |
| `src/data/eligibility.py` | Learns service/site/surgeon eligibility maps and checks whether a case can use a block. |
| `src/data/features.py` | Generic sklearn feature encoder for case features; the current estimation runner uses `src/estimation/quantile_model.py` directly. |

## `src/estimation/`

| Path | What it does |
| --- | --- |
| `src/estimation/__init__.py` | Defines the `EstimationResult` container. |
| `src/estimation/orchestrator.py` | Runs the full estimation pipeline in order. |
| `src/estimation/quantile_model.py` | Fits conditional quantile regressions for realized procedure duration. |
| `src/estimation/inverse.py` | Estimates surgeon critical ratios by matching booked times to predicted quantiles. |
| `src/estimation/response.py` | Estimates how surgeons adjust bookings after prior over/under-runs. |
| `src/estimation/profiles.py` | Clusters response parameters into profiles and produces SOS2 knots. |
| `src/estimation/recommendation.py` | Converts estimation output and weekly instances into data used by VFCG recommendations. |
| `src/estimation/bootstrap.py` | Runs surgeon-cluster bootstrap iterations for uncertainty intervals. |
| `src/estimation/serialization.py` | Saves and loads estimation artifacts with `joblib`. |

## `src/planning/`

| Path | What it does |
| --- | --- |
| `src/planning/__init__.py` | Public entry point for weekly instance building, evaluation, validation, and audits. |
| `src/planning/instance.py` | Converts a pool slice into a `WeeklyInstance` with cases, candidate blocks, and eligible blocks. |
| `src/planning/evaluation.py` | Evaluates realized schedule cost and KPI components. |
| `src/planning/audit.py` | Computes surgeon/day/site feasibility diagnostics. |
| `src/planning/schedule.py` | Legacy helper for parsing a solved Gurobi model into a `ScheduleResult`; current solvers use `ScheduleColumn.to_schedule_result`. |

## `src/solvers/`

| Path | What it does |
| --- | --- |
| `src/solvers/__init__.py` | Public entry point for optimization solver functions. |
| `src/solvers/deterministic.py` | Builds and solves the weekly deterministic MIP, pricing problem, and optimistic two-pass solver. |

## `src/methods/`

| Path | What it does |
| --- | --- |
| `src/methods/__init__.py` | Public entry point for scheduling methods and the registry. |
| `src/methods/base.py` | Abstract `Method` interface with `fit` and `plan`. |
| `src/methods/registry.py` | Ordered collection of methods for experiments. |
| `src/methods/booked.py` | Baseline that plans with booked durations. |
| `src/methods/oracle.py` | Diagnostic lower bound that plans with realized durations. |
| `src/methods/vfcg.py` | Main learned method: fits estimation, trains VFCG weights, and plans with post-review durations. |

## `src/vfcg/`

| Path | What it does |
| --- | --- |
| `src/vfcg/__init__.py` | Lightweight public entry point for VFCG config and result containers. |
| `src/vfcg/config.py` | Compatibility import for `VFCGConfig`, now defined with the rest of config in `src/core/config.py`. |
| `src/vfcg/types.py` | Result dataclasses for oracle calls, VFCG iterations, certification, and final training output. |
| `src/vfcg/solver.py` | Coordinates the VFCG loop: warm starts, master solves, oracle checks, and certification. |
| `src/vfcg/master.py` | Builds and solves the native VFCG master MIP. |
| `src/vfcg/oracle.py` | Exact follower oracle that prices schedules under candidate recommendation weights. |
| `src/vfcg/cuts.py` | Adds reference-schedule cuts and computes big-M constants for the master. |
| `src/vfcg/warmstart.py` | Generates initial reference schedules and seed weights. |
| `src/vfcg/certify.py` | Certifies the learned weights against follower-oracle violations. |
| `src/vfcg/diagnostics.py` | Logs compact per-iteration VFCG summaries. |
| `src/vfcg/init.py` | Compatibility placeholder. |

## `src/experiments/`

| Path | What it does |
| --- | --- |
| `src/experiments/__init__.py` | Public entry point for experiment orchestration. |
| `src/experiments/runner.py` | Runs the rolling-horizon experiment and writes experiment artifacts. |

## `scripts/`

| Path | What it does |
| --- | --- |
| `scripts/__init__.py` | Marks scripts as importable wrappers. |
| `scripts/data_analysis.py` | Large exploratory data analysis/reporting script. |
| `scripts/booking_realized_time_analysis.py` | Standalone booked-vs-realized time analysis with HTML/CSV outputs. |
| `scripts/paper_figures.py` | Generates paper-ready figures from data and analysis outputs. |
| `scripts/gap_runner.py` | Standalone two-plan gap runner comparing status quo/booked planning with oracle planning. |
| `scripts/estimation_report.py` | Exports estimation tables, checks, and quality diagnostics from saved artifacts. |
| `scripts/empirical_analysis.py` | Local targeted empirical analysis script; currently untracked in git. |
| `scripts/run_experiment.py` | CLI for the rolling scheduling experiment. |
| `scripts/run_estimation.py` | CLI for fitting estimation artifacts. |
| `scripts/run_estimation_report.py` | CLI for diagnostics from saved estimation artifacts. |

## `tests/`

| Path | What it does |
| --- | --- |
| `tests/conftest.py` | Test fixtures and path setup. |
| `tests/test_cli/test_entrypoints.py` | Checks that script command modules import. |
| `tests/test_core/test_artifact_paths.py` | Tests artifact directory/path construction. |
| `tests/test_core/test_column.py` | Tests schedule-column cost and conversion logic. |
| `tests/test_estimation/test_bootstrap.py` | Tests bootstrap behavior on small synthetic data. |
| `tests/test_estimation/test_inverse.py` | Tests critical-ratio estimation. |
| `tests/test_estimation/test_pipeline_integration.py` | Tests estimation pipeline, script report exports, and serialization together. |
| `tests/test_estimation/test_profiles.py` | Tests response profile clustering and SOS2 knots. |
| `tests/test_estimation/test_quantile_model.py` | Tests conditional quantile model fitting and prediction. |
| `tests/test_estimation/test_recommendation.py` | Tests recommendation data construction. |
| `tests/test_estimation/test_response_fit.py` | Tests fitted response parameters. |
| `tests/test_estimation/test_response_prep.py` | Tests response pair and residual preparation. |
| `tests/test_planning/test_audit.py` | Tests surgeon feasibility audit diagnostics. |
| `tests/test_vfcg/test_certify.py` | Tests VFCG certification logic. |
| `tests/test_vfcg/test_cuts.py` | Tests VFCG cut construction. |
| `tests/test_vfcg/test_master.py` | Tests native master-model behavior with small cases. |
| `tests/test_vfcg/test_method_integration.py` | Tests how `VFCGMethod` integrates with the experiment runner. |
| `tests/test_vfcg/test_oracle.py` | Tests the exact follower oracle. |
| `tests/test_vfcg/test_solver.py` | Tests VFCG loop coordination. |
| `tests/test_vfcg/test_warmstart.py` | Tests warm-start reference generation. |

## Local or Generated Folders

These are not part of the clean source design:

| Path | What it is |
| --- | --- |
| `artifacts/` | Generated experiment, estimation, analysis, and figure outputs. |
| `data/` | Local data files, including the UHN workbook and samples. |
| `presentation_figures/` | Generated or manually curated presentation images. |
| `venv/` | Local Python virtual environment. |
| `surgery_scheduling.egg-info/` | Packaging metadata from editable installs. |
| `__pycache__/` and nested `__pycache__/` folders | Python bytecode caches. |
| `tmp_*.py` and other `tmp*.*` files | Local experiments. |
| `src/bilevel/` | Currently only contains ignored bytecode cache files; no source module is present. |
