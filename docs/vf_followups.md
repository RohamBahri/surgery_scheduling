# One bounded VF follow-up campaign

Keep the paper-selection VF choice and the 2013 policies frozen. Run from
the repository root with the original experiment artifacts still present:

    git switch paper-selection-12h
    git pull --ff-only origin paper-selection-12h
    bash run_vf_followups.sh

Optional: CORES=10 bash run_vf_followups.sh

Source artifacts:
- artifacts/objective_selection_12h/objective_selection_2013/context.pkl
- artifacts/objective_selection_12h/objective_selection_2013/frozen.json
- artifacts/objective_selection_12h/single_split_2013/full_evaluation/per_arm/

New results go to artifacts/vf_followups/vf_followups/summary.json and
its stage directories. Original training, cached assignments and policies
are **not modified**. The launcher tests its mathematical and interface
contracts before executing. Stages are resumable, separately checkpointed,
and supervised with a maximum 11.5-hour controller, with stage ceilings
summing to 9 h 35 min: diagnose 15min, response 180min, tie 90min,
variant training 150min, variant evaluation 120min, duration heterogeneity
10min, report 10min. Time limits do not guarantee full exact coverage.

Experiments:

1. **VF slack:** for all existing certified 2013 assignments compute the
   exact fixed-schedule split identity U-g = (R-g)+(U-R), using 200
   scenario draws and their comonotone coupling. These are estimates
   from a scenario model, not observed joint dependence. The full
   primary-value decomposition also requires independent Q(d) certificates;
   do not equate U with Q+stoploss or assume eta=0.
2. **Response sensitivity:** recompute the scheduler under seven plausible
   fixed-display response scenarios for both frozen STAT and VF, including
   weak adoption and synthetic surgeon-specific profiles. No response
   parameters are estimated from historical bookings.
3. **Tie bracket:** selected outcome-blind components (the 16 largest and
   up to 48 sampled distinct-plan components); find best and worst
   25-scenario modeled cost inside the predicted-optimal tolerance set.
   Incomplete Gurobi solves retain their incomplete status. The headline
   fixed hash rule remains unchanged.
4. **VF ablations:** train congestion-only, no-congestion,
   no-interactions, and observed-duration instead of scenario-trained VF
   with the same STAT start and limited search; also rerun VF with
   the coefficient box expanded to +/-32. The observed-duration learner
   has no independent Bank B and is an overfitting diagnostic.
5. **Cross-site transfer:** retrain only the VF coefficient increments
   excluding TGH, TWH-main, or TWH-day, evaluate on the corresponding
   site's 2013 weeks. Original pooled CASE, STAT, feature normalization
   and residual histories remain in place, so this is *conditional
   increment transfer* and NOT truly leave-one-hospital-out validation.
6. **Surgeon duration heterogeneity:** compare global, raw surgeon-level
   and empirically shrunk log-error Gaussian predictive distributions
   out of period. Shrinkage strength is fixed at 20 previous cases.
   This estimates duration errors, not responses to unseen advice.

All new 2013 analyses are exploratory: this same retrospective cohort
has already been used to select VF. Do not claim a new independent test
or select the main method again based on these sensitivity results.
Summary statistics provide matched exact-coverage denominators and
paired calendar-week bootstrap intervals; missing components are never
silently treated as zero-cost or a win.

Robust-VF optimization is **not** in the initial campaign: the adoption
law has not been estimated from exposure to advice. Build robust VF
only if the frozen-response sweep shows a material sensitivity, and
label any parameter distributions as hypothesized scenarios until a
randomized display study or a valid natural experiment is available.
