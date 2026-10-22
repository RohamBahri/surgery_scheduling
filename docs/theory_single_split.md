# Full-train, chronological 2013 test (single split)

This is the **single-split** version of the full scheduling experiment. It
replaces the two outer development folds. It fits each method **once on all
24,506 pre-2013 cases** and evaluates frozen methods on the **8,774 cases
in 2013**. A four-way *inner* cross-fitting step remains to fit the residual
scenario model without in-sample residual bias; it is **not** two repeated
outer model fits or test-set model selection.

## Run

Stop the old `theory_full_v3` process first. Fetch branch
`theory-single-split-budgeted` and run `bash run_theory_full.sh`.
The launcher runs all theory tests, then the complete methods and full
2013 evaluation. Results are stored in a **new** directory,
`artifacts/theory_single_split_v1`, and must not be merged with
`theory_full_v3`. The runner requires `--acknowledge-final`, already
included by the launcher, because it opens the previously sealed 2013 test.

Only a *single* `single_split_2013` training/evaluation directory is
created. There is no `fold_0`, `fold_1`, or pooled development report.
The root `summary.json` contains the chronological design and the test
evaluation. Individual component JSONs remain available.

**Scientific limitation:** Opening the test set is one-time final evaluation.
Do not inspect it and then tune coefficients, hyperparameters, time budgets
or feature choices based on its outcomes. The result is retrospective,
not a deployment-identified causal effect.

## CPU allocation and bounded work

The default launcher allocates **15 cores total**. Independent scheduling
components are solved in parallel processes, one **Gurobi thread per worker**
to avoid oversubscription. The large coupled VF MIQP uses 15 Gurobi threads
in a *single* process. GLOBAL/ANCHORED shared-policy scoring also distributes
exact component deployments over a persistent 15-process pool. Independent
simulations and linear-algebra libraries use one thread inside each worker.

| Budget | Default | Scope |
| --- | ---: | --- |
| `SOLVE_SECONDS` | 90 s | One individual planner or MILP call |
| `TASK_SECONDS` | 600 s | One coefficient/case-component candidate shard |
| `ROUND_SECONDS` | 7,200 s | One directional search round, soft bound |
| `SEARCH_SECONDS` | 21,600 s | One directional learning method, soft bound |
| `PREPROCESS_SECONDS` | 7,200 s | Training SAA/target preprocessing pass |
| `GLOBAL_SECONDS` | 7,200 s | Each coefficient-region global search |
| `SHARED_SECONDS` | 14,400 s | Entire coupled VF model build and solve |
| `PW_SECONDS` | 480 s | Each per-instance optimization method |
| `EVAL_COMPONENT_SECONDS` | 4,200 s | Every tested component, all arms combined |

These are not a cap on the total experiment, because thousands of
independent components are evaluated. Budgets at the search-round level
are *soft*: already-running candidate shards finish within their own
600-second bound. Model-build budgets are polled between components,
not by an operating-system process killer. A Gurobi solver itself
receives the available remaining time.

The five-component features, same response maps, all methods, 50/50/200
scenario banks, three deterministic hash seeds, and whole-cohort
denominators are unchanged. This is **not a toy or reduced-case run**.

## Progress preservation and failure semantics

Exact deterministic planner assignments are cached after all requested
hash seeds are certified; incomplete results are never cached as exact.
Candidate tasks are spread across workers in shards (especially large
components), and completed shards are saved on arrival. Directional
search checkpoints its last independently confirmed policy after each
complete round. A timeout never causes an incomplete candidate to
be accepted. If a search cannot certify a new candidate, its previous
confirmed coefficient vector remains available and the trained JSON
records `BUDGET_EXHAUSTED` or a more specific status.

All SAA preprocessing tasks are checkpointed. If a task has no certified
dual lower bound, it contributes zero, a **valid but weaker** bound on
the nonnegative reduced scheduling cost. Missing restricted target
information stops only ANCHORED_TARGET, which is reported unresolved
instead of inventing a target. The global coefficient search
checkpoints its queue and feasible incumbent; completed fitted
solutions are frozen before test evaluation. VF stores a feasible MIP
start, **not** an entire restorable Gurobi search tree.

For each 2013 component, the evaluator attempts all named methods
under the 4,200-second component budget. Each expensive per-week
method also has its own 480-second sub-budget. If a method times
out *with* a feasible advice vector, that vector is independently
deployed/evaluated and its unfinished optimization certificate is kept.
If there is no feasible advice vector, the arm is reported
`UNRESOLVED`, never secretly replaced by CASE. The global
reported case-weighted headline mean is withheld for an arm if
even one eligible component is unresolved. All solver timings,
unresolved counts, SAA bounds, inducibility intervals and
site/size subgroup information are retained.

A modeling error, violated lower bound or corrupt checkpoint is
**not** treated as a timeout: the runner fails closed instead of
silently publishing mathematically inconsistent results.

## Scope of the experiment

The fitted methods are SHARED, OBS, EG, ROBUST, globally bounded
SHARED, ANCHORED_TARGET and the globally bounded five-coefficient
VF objective. The test comparison includes BOOKED, CASE, STAT, RISK,
QFILL, VF per-instance, PER_WEEK, PER_WEEK_ROBUST, bounded unrestricted
advice, target-restricted advice, the seed-0 value-function-cut model,
and the nondeployable unrestricted SAA reference.

EG remains an **exact inner adversarial MILP** within an
approximately optimized outer objective. The per-instance value-function
solver has *necessary relaxed hash cuts*, so it only claims global
optimality when a verified deployed incumbent closes a valid dual bound.
The SAA lower bound ignores advice inducibility. Expected model scores
are not equated with real-world outcomes.

## Useful commands

```bash
git fetch origin
git switch theory-single-split-budgeted
git pull --ff-only origin theory-single-split-budgeted
bash run_theory_full.sh
```

Change only environment variables to configure budgets *before* the
final evaluation, such as
`CORES=12 SEARCH_SECONDS=14400 EVAL_COMPONENT_SECONDS=5400 bash run_theory_full.sh`.
Any changed configuration requires a **new output directory**.
The manifest hashes all scientific dependencies, the workbook,
method list, and budgets to prevent mixing incompatible checkpoints.

Check status by inspecting the newest JSON files below
`artifacts/theory_single_split_v1/single_split_2013`.
The final report is `theory_summary.json` there, plus root
`summary.json`. No background service or automated upload is implied.
