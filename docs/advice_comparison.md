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

For a *single* frozen 2013 evaluation, after development settings are final:

```bash
python pilot/run_advice_comparison.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/advice_comparison --cores 15 --final-2013
```

This refits on all 24,506 pre-2013 cases and evaluates the 8,774 2013 cases once. Do not change the model or inspect/reuse the final outcomes for selection.

## Methods

- **BOOKED**: original bookings.
- **CASE**: fitted, response-aware CASE baseline (inverse link).
- **RISK**: casewise 0.636 conditional quantile, without planner optimization.
- **STAT**: CASE plus five features trained by asymmetric duration loss.
- **SHARED**: same five features with coefficients additionally selected to reduce scenario scheduling cost.
- **PER_WEEK**: per-component coordinate and transition search using the deployed tie rule.
- **PER_WEEK_ROBUST**: same per-component search, averaged over three fixed tie rules.

The five features are historical surgeon/procedure absolute booking error, the CASE baseline room congestion averaged over three tie rules, and its interactions with those two history measures. They are standardized on each fitting fold only. Method B uses a separate recommendation for every current case, bounded by the nominal response-feasible interval. Its search is heuristic; certified *planner* selection does not imply global prescription optimality.

The scenario model draws **50** out-of-fold log-duration residuals per bank from the 30 nearest predictions for the same surgeon. Bank A chooses candidates; independent B confirms; independent C reports model-based results after selection. B is a selection bank, not an unbiased final estimate. Fold outcomes and the frozen 2013 replay, not bank B, determine empirical evidence. The fitted distribution assumes independent case residuals and may be misspecified.

**Performance**: All policies are evaluated on the identical held-out cases and on each of hash seeds 0–2. `summary.json` contains paired cost-per-case differences vs CASE and calendar-Monday bootstrap intervals. `components/*.json` records assignments, displays, acceptance/fallback status, model-bank costs and retrospective cost. `pooled.json` combines both development folds. An uncertified CASE baseline aborts the experiment; an uncertified optional recommendation falls back to CASE and records its status. No uncertified plan is treated as an optimal schedule.

**CPU**: exactly `--cores` spawned worker processes (maximum 15), one Gurobi thread per process; numerical/BLAS pools in workers are capped to one thread. Sequential CASE/OOF fitting permits up to the supplied 15 BLAS threads. Do not run another multi-core Gurobi job concurrently; multiple active solves also require a suitable Gurobi license. Output files and frozen-settings checks support restarting after an interruption.

## Interpretation

The planner objective scores idle and overtime against 480-minute physical room-days and 30-minute turnovers; it counts every roster slot, including unused slots. The analysis assumes surgical durations do not depend on assignment and that the reconstructed roster and surgeon eligibility were available at the planning cutoff. Surgeon response parameters (0.8, 30) are **assumptions**, because displayed-advice responses were not observed in the hospital data. A superior scenario-bank score does not establish superior realized cost.
