# Clean all-site anchored experiment

## Scope and protocol

This branch reuses the frozen UHN rich-feature matrices and read-only
CASE, LASSO, VF, and VF-GAP coefficients. It DOES NOT reuse any old anchored
learning checkpoints. The default NEW output directory is
artifacts/anchored_ALL_hash_v2.

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

The strengthening preserves all feasible integer assignments. The deployed
planner minimizes predicted idle/overtime cost, then, among assignments with
cost at most the certified minimum plus the predeclared numeric tolerance,
minimizes a fixed integer SHA-256 case/physical-room-day hash score (seed 0).
It NEVER minimizes historical case moves, prefers historical room-days, or
balances peak load as a secondary objective. Actual durations never enter a
planning objective. Physical rooms and historical required surgeon-days remain
part of the feasible set, not the objective.

Primary MIPs use a limited portfolio of default/focused Gurobi configurations
with fixed, recorded search seeds. Gurobi concurrent MIP shares the threads
allocated to a worker; workers * threads may not exceed the configured core
budget. Any unfinished primary objective, hash minimization, or deterministic
selection is reported incomplete, never evaluated as an exact deployment.

Hash arc weights are stable integers in [1, 2^20]. After certifying the first
hash minimum, a no-good exclusion and integer cutoff checks whether its winning
assignment is unique. In case of an actual collision the solver minimizes a
second fixed hash, subject to the first score, and finally uses exact ordered
case-room lexicographic selection if both hash scores collide. If uniqueness
cannot be certified within the budget, deployment stays INCOMPLETE. A certified
workload conservation bound may skip ONLY the primary MIP; it never bypasses
hash tie selection unless there is a unique feasible assignment.

The primary cost tolerance does not disappear under hashing. The regret
certificate continues to include gamma times the certified upper bound on
deployed primary-cost deviation delta. Do not round durations to speed solving.
Each deployed result stores primary/tie/uniqueness status, primary search
configuration and Gurobi seed, fixed hash seed, scores, bounds and stage times.


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
    bash run_all_sites_anchored.sh sensitivity

The runner uses 15 one-thread worker processes for independent weekly MIPs,
four Gurobi threads for the sequential pDCA master QP, and a 600-second
per-solve main budget. The cut/master budgets, hash seed and the number
of iterations are pinned in run_all_sites_anchored.sh. The --dry-run flag
shows the exact command without executing the solver.

Run only in a NEW output directory. Old movement-rule artifacts are pilot
records and are incompatible with this experiment. The hash-seed setting is
included in the run identity; do not tune it on validation outcomes.
 The anchored_settings.json identity
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

## Interpretation and diagnostic checks

The hash is an outcome-independent deterministic selector, not a uniform random
schedule and not a claimed regret-minimizing secondary objective. The previous
historical movement rule was removed because the workbook supplies performed
rooms rather than a verified prior room plan. Do not select a tie rule because
it produces a more favorable comparison. The old-rule pilot is exploratory.

The final 2012 development study must report paired realized cost/regret across
BOOKED, CASE, LASSO, VF, VF-GAP, and ANCHORED with seed 0. Evaluate frozen policy
sensitivity under prespecified other hash seeds separately, without choosing
or refitting a favorable seed. Audit physical room-day loads above 720, 960,
and 1440 minutes before concluding that the modeled mobility is realistic.
To inspect the frozen ANCHORED model on training seeds 1--4, use
`python3.14 -m legacy.anchored.run_weekly_anchored ... --stage sensitivity --sensitivity-split train`
with the same pinned workbook/artifact/solver flags as the all-site runner.
Each run writes seed-specific resumable CSVs and a separate summary; no
training coefficients or acceptance decisions change. Evaluate seed 0 first.
The 2013 holdout remains untouched until all decisions are frozen.
