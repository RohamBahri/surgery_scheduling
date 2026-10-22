# Experiment entrypoints: current VF vs. historical code

**Canonical branch:** `main`. **Headline learner:** VF. The October 2026 selection and follow-up experiments are completed and must not be rerun just to choose a different method.

## Active research record (keep and maintain)

| File | Purpose | Status |
|---|---|---|
| `run_selection_12h.sh` | Launch the frozen 2013 method-selection campaign | Reproduction only |
| `run_vf_followups.sh` | Launch VF ablations, slack diagnostics, response sensitivity, tie checks and transfer analyses | Reproduction only |
| `pilot/run_objective_selection.py` | Selection stage CLI and report generation | Current |
| `pilot/run_vf_followups.py` | VF follow-up stages and report generation | Current |
| `pilot/run_theory_full.py` | Common primary planner, scenario scoring and VF evaluation engine | **Required library**; keep |
| `surgery/objective_selection.py`, `surgery/theory_exact.py`, `surgery/anchored_weekly_solver.py` | Training and numerical solver foundations | **Required libraries**; keep |

The module `surgery/anchored_weekly_solver.py` is the common physical-room solver used by VF. Its historical name does not identify the paper's learning method. Likewise, `pilot/run_theory_full.py` is imported by current programs: it is **not safe** to remove simply because it looks like an old run script.

The study uses `artifacts/objective_selection_12h/` and `artifacts/vf_followups/` (ignored by Git); their checkpoints have source fingerprints. Reproducing with scientifically different code requires a new artifact directory. The 2013 comparison has already been inspected repeatedly and is not independent external validation.

## Historical numerical entrypoints (retain for traceability)

These implement obsolete *research experiments*, not current paper alternatives. **Do not launch without a specific historical reproduction goal.**

| Family | Entrypoints | Retention reason |
|---|---|---|
| Early daily and weekly planning | `run_weekly.py`, `run_weekly_diagnostics.py` | Called by regression tests for scheduling and reproducibility |
| Weekly feature and case training | `run_weekly_rich.py`, `run_weekly_train.py` | Imported by current research dependencies and tests |
| Weekly GAP and anchored study | `run_weekly_gap.py`, `run_weekly_anchored.py` | Used by legacy research and regression tests |
| Anchored diagnostics | `run_anchored_extras.py`, `run_anchored_research_diagnostics.py` | Imported by retained tests |
| Folded and frozen pilots | `pilot/run_pilot.py`, `pilot/run_fold_pilot.py`, `pilot/run_frozen_validation.py` | Dependencies and earlier experiment audit |
| Retrospective 2013 pilot | `pilot/run_final_2013.py`, `pilot/audit_final_2013.py` | Earlier experiment and provenance |
| ASDBS and advice comparison | `pilot/run_asdbs.py`, `pilot/run_advice_comparison.py` | **Imported by the current VF evaluation stack**; keep |
| Historical shell pipelines | `pilot/run_fold_pilot.sh`, `pilot/run_frozen_validation.sh`, `pilot/run_final_2013.sh` | Reproduction recipes for historical projects, not actively supported |

Some historical scripts import one another directly by their original module names and are checked by tests. Moving them into a different package or deleting them is a **refactor**, not housekeeping. Avoid it unless accompanied by reference updates and complete CI validation.

## Local-only files and cleanup

The committed `.gitignore` excludes local data, experiment artifacts, `.DS_Store`, `*.tgz`, `gurobi.env`, and generated `presentation_figures/`. It deliberately does **not** exclude an unknown `src/` directory or untracked `*.py` research scripts: review those before hiding them.

Move confirmed scratch code to the ignored `local/` directory rather than deleting it. Do not use `git clean -fdx`: this could permanently remove ignored input data, results, and license files.

### Historical branch provenance

Remote branches have already been retired, leaving `main`. Verified annotated tags retain the frozen VF source and diverged W2B history:

- `research/vf-followups-frozen-2026-10-22`
- `archive/w2b-weekly-surrogates-2026-10-22`

Local branches may still exist after remote deletion. Inspect `git branch -vv` and use `git branch -d <branch>` only for branches merged into main. Keep any locally divergent branch until its unpublished commits are reviewed.
