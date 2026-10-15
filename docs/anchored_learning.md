# Clean all-site anchored experiment

## Scope and protocol

This branch reuses the frozen UHN rich-feature matrices and read-only
CASE, LASSO, VF, and VF-GAP coefficients. It DOES NOT reuse any old anchored
learning checkpoints. The default NEW output directory is
artifacts/anchored_ALL_clean_v1.

One shared 74-feature correction policy is trained on 18,705 cases across
180 TGH, TWH-main, and TWH-day site-weeks, starting from frozen CASE weights.
Development validation uses 5,779 cases in 54 later site-weeks (18 calendar
weeks). Do not access the 2013 holdout for tuning or model selection.

## Exact optimizer

The anchored experiment uses the single independent strengthened Gurobi
solver in surgery/anchored_weekly_solver.py across ALL call sites:
references, deployment verification, tangent separation, regret certificates,
and validation. The original surgery/weekly_planner.py remains unchanged to
preserve the source fingerprint of the frozen rich-feature matrices.

The strengthened model adds the valid inequalities

    x[i,j] <= used[j]     for every eligible case-room assignment arc,

and fixes used[j] = 1 only when original surgeon-day retention implies that
the physical room-day must be opened in EVERY feasible integer schedule.

This does not change the set of feasible integer assignments. The optimizer
still minimizes predicted room idle/overtime, then the number of moved cases
within the original primary cost tolerance, then the lexicographic ordered
case-to-room assignment. Realized durations never enter deployment.

Two optional exact shortcuts apply only with a mathematical certificate:
a workload-conservation lower bound when all slots are forced open, and
the original zero-move schedule if its cost lies strictly inside the
certified primary-optimum tolerance. Otherwise all optimization stages are
solved by Gurobi. There are no general room symmetry-breaking assumptions.

## Training correctness

Hindsight cost for EVERY component must be certified by an exact primary
optimization. Reference proposals are feasible schedule aspirations,
not necessarily optimal deployments. If an OPTIONAL reference-policy
optimization times out, its feasible incumbent can be used as a reference;
if none exists, the original feasible assignment can be used. These rows
are marked "uncertified-feasible-aspiration" and have
reference_optimality_certified=false. Never present these rows as the
deployed baseline schedule or as certified baseline performance.

Policy acceptance and validation require COMPLETE exact deployed-policy
schedules. A time-limited trial is logged as unverified and cannot be
accepted; the training loop may try other step fractions. If the current
policy is unverified, training stops. No week/site is silently dropped.

Adaptive schedule separation and pDCA remain finite-budget heuristics;
realized training improvements do not establish global bilevel optimality,
surrogate closure, or out-of-sample improvement. Final certification and
paired development-validation evaluation are reported separately.

## How to run

Update the anchored-exact-adaptive branch, then run:

    python3.14 -m pytest -q
    bash run_all_sites_anchored.sh references
    bash run_all_sites_anchored.sh train
    bash run_all_sites_anchored.sh certify
    bash run_all_sites_anchored.sh evaluate

The runner uses 15 one-thread worker processes for independent weekly MIPs,
four Gurobi threads for the sequential pDCA master QP, and a 600-second
per-solve main budget. The first three cut/master budgets and the number
of iterations are pinned in run_all_sites_anchored.sh. The --dry-run flag
shows the exact command without executing the solver.

Run only in a NEW output directory. The anchored_settings.json identity
binds source code, data, baseline hashes, and optimizer hyperparameters;
incompatible restarts are rejected. Finished work is saved in checkpoint
JSON files and can be resumed with the same commands and settings.

Evaluate development validation only after training is frozen. Compare
BOOKED, CASE, LASSO, VF, VF_GAP, and the proposed ANCHORED policy on the
same 54 site-weeks. Report paired calendar-week bootstrap intervals.
The 2013 holdout remains untouched.

Primary outputs:
- anchored_references.json: certified hindsight and labeled feasible proposals.
- anchored_progress.json: accepted training outcomes, unresolved trial records.
- anchored_verify_*.json: cached exact deployed schedules.
- anchored_exact_separation_*.json: adaptive exact tangent cuts and dual audits.
- anchored_bound_summary.json: conditional final training regret certificates.
- anchored_validation.csv / anchored_validation_summary.json: paired validation.

Temporary source-migration and reference-rescue entry points used during
the exploratory run have been retired. Git history preserves them.
