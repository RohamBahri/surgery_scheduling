# Final paper experiment protocol — superseded

This file is retained only to prevent old links from breaking.

**Do not run `run_final_paper_training.py`, `run_final_paper_evaluation.py`, or the legacy experiment scripts directly.** Those implementation modules do not, by themselves, guarantee that every reviewed hardening layer is installed.

The operational source of truth is:

- `docs/final_experiment_protocol.md`
- the single supported command wrapper: `python run_final_paper.py ...`

The current workflow solves the behavior-independent training oracle and BOOKED
weekly plans once with `python run_final_paper.py shared-plans ...`, then trains
one or more explicitly declared `(alpha,h)` regimes with
`python run_final_paper.py train --shared-plans-root ... --alpha ... --h ...`.
Use `python run_final_paper.py evaluate ...` for the one-shot holdout and
`python run_final_paper.py sensitivities ...` for required structural checks.
