# External-review reconciliation — single-split theory experiment

This file distinguishes **implemented safeguards**, **mathematical differences
that cannot be eliminated by changing optimizer flags**, and **research
extensions not claimed by this run**.

## Implemented in the audited branch

| Review issue | Current treatment |
|---|---|
| Redundant outer development folds | Removed. Fit once on 24,506 pre-2013 cases; evaluate 8,774 later cases. Inner OOF fitting for residual nuisances remains. |
| Large-component stragglers | Candidate vectors are broken into small independent shards, down to one candidate for large components/ROBUST; 15 one-thread Gurobi workers. |
| Unbounded nested computation | Per-solver, shard, round, method, preprocessing, global MIP, per-component and mechanism deadlines. Missing solver certificates remain unresolved; code proceeds with the last feasible incumbent. |
| VF objective/optimization | One coupled globally bounded MIQP; full fixed assignment QFILL and per-instance VF remain. VF runs before directional training and supplies an objective-verified optional initialization. |
| Limited 8/2^round coordinate search | Coefficient steps begin at radius 32, candidates are clipped to the declared coefficient box, and exact coordinate boundary proposals are included. This improves reach but does NOT make directional search global. |
| OBS Bank-B not independent | Removed its fake confirmation. OBS uses full fitting-outcome improvements, with independent later test evaluation only. Scenario A/B for other methods remain independent *draw banks*, not a formal adaptive statistical guarantee. |
| Primary cut caching | Reuse of a fixed-schedule cost expression and installation of the necessary follower-optimality inequality are separately tracked. |
| VFCG seed mismatch | Uses three leader schedules, common reviewed advice and a shared primary witness. Its valid lower bounds and verified incumbent cost refer to the same three-seed averaged objective. Strict hash boundaries remain relaxed via necessary nonstrict cuts. |
| Changed pw_nodes and stale checkpoints | Included in the content-hashed manifest, along with source, data and current time budgets. Separate v1 artifact directory. |
| Dependence-model error | Independently sampled Bank C and a comonotone rearrangement with **identical empirical one-case marginals** are evaluated for each saved assignment. This is a fixed-policy sensitivity, not a retrained distributionally robust optimization. |
| Tie-rule dependence | The exact hash selector is still the primary experiment. Peak-load balancing is evaluated as a distinct diagnostic for core arms, not substituted into main comparisons. |
| Unidentified behavior | Response sweep evaluates fixed displays under (0.5,30), (0.8,15) and (0.5,15), with the nominal (0.8,30) as reference. These are scenarios, not learned surgeon effects. |
| Timing misleadingly omitted PER_WEEK fitting | Full per-method optimization-plus-deployment wall time is recorded separately from pure test deployment time; all failed/unfinished arms retain coverage counts. |
| Long interrupted work | Exact-deployment SQLite cache, completion-order shard records, completed search rounds, SAA preprocessing, saved global coefficient queue, and VF MIP-starts; VF's internal Gurobi tree is NOT fully restorable. |
| Scenario/realized comparability | Every successfully deployed method is assessed under the same Bank C and recorded historical durations; SAA is kept separately as nondeployable. Model costs, realized outcomes and bound certificates are not conflated. |

## Deliberately *not* declared resolved

1. **Distribution identification.** The 50/50/200 residual resampling
   banks are a specified uncertainty model, not an identified joint
   predictive distribution. Holding marginals fixed and changing dependence
   is diagnostic; the training law is not automatically corrected.
2. **Global optimizer quality.** EG's finite penalty and VF's stop-loss
   envelope are different objectives. A tighter bound does not imply a
   better deployed recommendation; a method's solver gap matters. Direct
   coefficient-region bounds can be weak under dense hash ties.
   The proposed selection-aware primary-value interval certificate and
   tighter EG-specific global region bounds have **not** been implemented.
   No claim is made that a budget-limited method is globally optimal.
3. **Exact inducibility.** VFCG generates necessary hash-disjunction cuts,
   with strict inequalities relaxed. Certification occurs only if a
   verified feasible policy closes the valid lower/upper bounds. Its
   target-ANCHORED counterpart is a restricted collection of target
   schedules, NOT an exact unrestricted induction theorem. Version 4's
   fixed-margin target-selection argument remains invalid.
4. **Review mechanism decomposition.** The proposed three nonnegative
   pieces separating VF, finite-penalty EG, worst-near-tie and actual
   hash outcome have not been computed via an exhaustive worst-tie oracle
   on the full cohort. The report does provide assignment regimes,
   hash disagreement, comonotonic stress, and balanced-tie outcomes.
5. **Operational realism.** Original case-to-room eligibility and
   retention constraints are unchanged. There is no full surgeon
   nonoverlap constraint or risk-calibrated nonlinear overtime penalty.
   Per-component near-tie tolerances differ from a globally coupled
   weekly selection policy. Evaluation-week roster reconstructions can
   exploit retrospective performed-record evidence. We do not present
   the retrospective planner as a fully prospective clinical deployment.
6. **Information cutoff sensitivity.** Alternative administrative
   cutoffs cannot be honestly reconstructed without a prospective
   availability/roster feed. No ungrounded synthetic claim is inserted.
7. **Statistical independence.** Model seeds and multiple hash rules do
   not multiply the number of independent hospital weeks. Weekly
   bootstrap inference is exploratory and does not prove population
   improvement after adaptive model development.
8. **Test exposure.** Historical project work has already inspected 2013
   performance. A single chronological split is the requested
   experiment, but its outcomes must be described as a retrospective
   previously exposed test, not a pristine sealed prospective test.

## Core result interpretation

The policy classes use a matched five-feature interface and response
bounds but optimize **different objectives**. Summaries must compare
the same deployed cost, model-C cost and feasible roster across methods.
The unrestricted SAA reference is only an optimization lower bound
for realized advice and is not necessarily attainable. A direct
PER_WEEK feasible policy is an upper bound on best inducible cost, not
its exact optimum unless certified.

The branch intentionally fails on contradictory primal/dual bounds,
invalid data partitions or corrupted caches. Such a failure is a
scientific correctness issue, not a harmless wall-time expiration.
A wall-time expiration should retain a valid feasible policy or
an explicit unresolved status and continue with later work.
