# Surgery Scheduling

This repository contains the current experiment stack for incentive-aware OR scheduling.

## Current status
- Implemented: rolling-horizon evaluation with `Booked` and `Oracle` methods plus estimation and diagnostics pipelines.
- Legacy: the CCG-based behavioral method remains in `src/methods/legacy_behavioral_ccg.py` as a **legacy heuristic** (`LegacyBehavioralCCGHeuristic`), not the document's exact VFCG solver.
- Next stage (not implemented here): exact VFCG method integration.

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
