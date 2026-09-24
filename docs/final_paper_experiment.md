# Final paper experiment protocol — superseded

This file is retained only to prevent old links from breaking.

**Do not run `run_final_paper_training.py`, `run_final_paper_evaluation.py`, or the legacy experiment scripts directly.** Those implementation modules do not, by themselves, guarantee that every reviewed hardening layer is installed.

The operational source of truth is:

- `docs/final_experiment_protocol.md`
- the single supported command wrapper: `python run_final_paper.py ...`

For Stage 1 use `python run_final_paper.py train ...`; for the one-shot holdout use `python run_final_paper.py evaluate ...`; and for the required structural checks use `python run_final_paper.py sensitivities ...`.
