# Surgery Scheduling

This repository contains the current experiment stack for incentive-aware OR scheduling.

## Current status
- Implemented: rolling-horizon evaluation with `Booked` and `Oracle` methods plus estimation and diagnostics pipelines.
- The production learning method is now the exact VFCG stack under `src/vfcg/` with the compact weekly MIP as its only follower oracle.
- The CLI and experiment runner include VFCG training diagnostics and certification outputs.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI commands
```bash
or-run-experiment --help
or-run-estimation --help
or-run-estimation-report --help
or-run-data-analysis --help
```

## Artifact policy
All generated outputs are written under one root (default `artifacts/`):

```text
artifacts/
  experiments/<run_id>/
    config_snapshot.json
    scope_summary.json
    eligibility_summary.json
    horizon_results.csv
    aggregate_summary.json
  estimation/<run_id>/
    estimation_artifacts.joblib
    diagnostics/
  analysis/<run_id>/
    analysis_report.txt
    figures/
    tables/
```

Use `--artifact-root` and optional `--run-label` on each CLI to control output location.
