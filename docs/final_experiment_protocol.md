# Experiment protocol — moved

The previous single-scenario protocol in this file is superseded.

Use **`docs/experiment_protocol.md`** as the operational source of truth. The
current protocol freezes the committed multi-scenario registry, reusable
Oracle/BOOKED training backbone, cross-scenario comparability audit, pre-holdout
experiment seal, common full-budget response matrix, experiment-level holdout
cache, exact-input restart rule, and the reviewed numerical-accounting policy.

The only supported experiment command remains:

```bash
python run_final_paper.py ...
```

Do not use the implementation-stage scripts directly for the paper experiment.
