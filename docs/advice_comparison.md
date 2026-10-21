# Behavioral advice comparison (full UHN cohort)

New experimental branch; legacy estimators, planner, and previous experiments are not edited.

## Run

Install the repository requirements and provide the UHN workbook locally.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python pilot/run_advice_comparison.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/advice_comparison --cores 15
```

Default: both parity-of-Monday folds across **all 24,506 pre-2013 cases**; no large-component exclusion. CASE is refitted on the opposite fold, with four inner calendar-week refits for out-of-fold duration residuals. Each held-out site-week is partitioned into the exact independent room-day eligibility components. Every algorithm uses the same unchanged strengthened MILP and fixed hash tie rules. Models see bookings, historical features, and retrospectively reconstructed roster data; never the evaluated cases' realized durations when choosing advice.

For a **repeatable, versioned** 2013 evaluation, ideally after development choices are final:

```bash
python pilot/run_advice_comparison.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/advice_comparison --cores 15 --final-2013
```

This refits on all 24,506 pre-2013 cases and evaluates the 8,774 2013 cases. Runs with different code/settings are saved separately under `final_2013/runs/<hash>/`, with `final_2013/summary.json` pointing to the latest result. There is no one-time lock. Repeated evaluation is possible, but tuning on the 2013 outcomes undermines claims of a held-out test.

## Methods

- **BOOKED**: original bookings.
- **CASE**: fitted, response-aware CASE baseline (inverse link).
- **RISK**: casewise 0.636 conditional quantile, without planner optimization.
- **STAT**: CASE plus five features trained by asymmetric duration loss.
- **SHARED**: same five features with coefficients additionally selected to reduce scenario scheduling cost.
- **PER_WEEK_REF**: coordinate-only search for components up to 20 cases; CASE for larger components. This restricted comparison resembles the earlier pilot but does not assert numerical reproduction before running.
- **PER_WEEK**: identical coordinate search through 20 cases, and transition-guided search on larger components using hash rule 0; it may cross up to three non-improving intermediate boundaries before accepting a bank-A improvement.
- **PER_WEEK_ROBUST**: same size-dependent search, with three fixed hash rules averaged for its search and confirmation. We do not change the planner's hash tie implementation.

The five features are historical surgeon/procedure absolute booking error, the CASE baseline room congestion averaged over three tie rules, and its interactions with those two history measures. They are standardized on each fitting fold only. Method B uses a separate recommendation for every current case, bounded by the nominal response-feasible interval. Its search is heuristic; certified *planner* selection does not imply global prescription optimality.

The scenario model draws **50** out-of-fold log-duration residuals in banks A and B, and **200** in bank C from the 30 nearest predictions for the same surgeon. Bank A chooses candidates; independent B confirms; independent C reports model-based results after selection. B is a selection bank, not an unbiased final estimate. Fold outcomes and the frozen 2013 replay, not bank B, determine empirical evidence. The fitted distribution assumes independent case residuals and may be misspecified.

**Performance**: All policies are evaluated on the identical held-out cases and on each of hash seeds 0–2. `summary.json` contains seed-specific comparisons and the **average across seeds as a single paired outcome**, with 20,000 calendar-Monday bootstrap resamples and 95% confidence intervals. The two predeclared primary comparisons are P1: PER_WEEK minus CASE and P2: SHARED minus CASE, with two-sided bootstrap p-values adjusted by Holm. Other contrasts include ROBUST minus PER_WEEK, SHARED minus STAT, PER_WEEK minus SHARED, and PER_WEEK minus RISK. Pooled results include both folds and site/size strata. `components/*.json` records assignments, displays, acceptance/fallback status, model-bank costs and retrospective cost. `pooled.json` combines both development folds. The CASE and BOOKED baselines (and training CASE feature schedules) are given at least 600 seconds per solve; an uncertified CASE still aborts the experiment with component/seed identification rather than silently dropping cases. Uncertified optional recommendations fall back to CASE and record their statuses. No uncertified plan is treated as an optimal schedule.

**CPU**: up to `--cores` spawned worker processes (maximum 15); concurrent Gurobi solves receive one thread each for large batches, while small batches distribute spare threads to their sequential solves. BLAS pools in workers are capped to one thread. Sequential CASE/OOF fitting permits up to the supplied 15 BLAS threads. Do not run another multi-core Gurobi job concurrently; multiple active solves also require a suitable Gurobi license. Output files and source-dependent cache fingerprints support restarting after an interruption. Existing development caches reject a code/data change; use a fresh development output directory after updating the code.

## Interpretation

The planner objective scores idle and overtime against 480-minute physical room-days and 30-minute turnovers; it counts every roster slot, including unused slots. The analysis assumes surgical durations do not depend on assignment and that the reconstructed roster and surgeon eligibility were available at the planning cutoff. Surgeon response parameters (0.8, 30) are **assumptions**, because displayed-advice responses were not observed in the hospital data. A superior scenario-bank score does not establish superior realized cost.


## Frozen-display sensitivity (optional, after completing both development folds)

```bash
python pilot/run_advice_comparison.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/advice_comparison --cores 15 --sensitivity-only
```

No new recommendations are learned. For each stored displayed advice, the experiment applies response configurations `(alpha, h) = (0.5,15), (0.5,30), (0.8,15), (0.8,30)`, solves the same exact planner for each hash rule, and scores the same realized durations. **Changing h from 30 to 60 would have no effect** on displays that were already capped at 30, hence the lower h sensitivity. A separate `cutoff_week_start` replay gives an arm's frozen advice only to cases whose recorded `Decision_Date` is *before Monday* of the surgical week; all others receive CASE advice. This is a retrospective cutoff proxy, not a causal simulation of arrivals or a historical proof of which data a real scheduler knew. Missing decision dates are explicitly counted and treated as not known. Output is versioned under `sensitivity/runs/<hash>/` with a pointer `sensitivity/latest.json`. Because every response perturbation can change planner choices, this stage may require many additional certified MIP solves.

The development baseline's exact-planner tie rule and all roster assumptions remain unchanged. No global prescription optimality or oracle lower-bound certificate is claimed.
