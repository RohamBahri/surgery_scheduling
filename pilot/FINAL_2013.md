# Full pre-2013 training, fast audit and repeatable 2013 evaluation

This run uses **all 24,506 cleaned pre-2013 cases for fitting** (including the former 2012 validation weeks) and the separate **8,774 2013 cases for evaluation**. There is no additional validation partition. It keeps the same inverse-response assumptions (`alpha=.8`, `h=30`) and cohort features as the earlier pilot.

The established primary comparison remains **ANCHORED (ridge .001) versus cohort CASE**. BOOKED, case-only CASE, cohort LASSO, ANCHORED ridge .003 and VF are secondary. VF-GAP is **optional and exploratory**. Model settings are not changed based on past 2013 outcomes automatically; you may rerun or analyze 2013 however you wish, but repeated test-guided choices should be disclosed if presenting confirmatory inference.

## Important correction: training response link

A prior version **fitted CASE, ANCHORED and VF with the original surgeon-response link** because it set `SURGERY_RESEARCH_LINK=inverse` only during test scoring. Such policies are not valid inverse-link training outputs. The new runner sets the inverse link before **every** stage and records it in `experiment_settings.json`. It rejects any prior root with missing or mismatching link metadata.

The default output root is now `artifacts/final_2013_inverse_v2`. **Do not transfer CASE policy files, ANCHORED references or checkpoints, VF, or GAP from `artifacts/final_2013_audited_v1` into it.** Those must be retrained under the corrected link. Existing exploratory test folders are preserved. You may rerun 2013 evaluations whenever you wish.

To save feature-building time, the following **link-independent inputs only** can optionally be copied from the interrupted old root to the fresh root *before* training:

```bash
OLD=artifacts/final_2013_audited_v1
NEW=artifacts/final_2013_inverse_v2
mkdir -p "$NEW/none" "$NEW/cohort"
for f in training_data.npz training_weeks.pkl feature_encoder.pkl training_manifest.json; do
  if [ -f "$OLD/$f" ] && [ ! -e "$NEW/$f" ]; then cp "$OLD/$f" "$NEW/$f"; fi
done
for ctx in none cohort; do
  if [ -f "$OLD/$ctx/fold_lasso.json" ] && [ ! -e "$NEW/$ctx/fold_lasso.json" ]; then
    cp "$OLD/$ctx/fold_lasso.json" "$NEW/$ctx/fold_lasso.json"
  fi
done
if [ -f "$OLD/cohort/context_scaler.json" ] && [ ! -e "$NEW/cohort/context_scaler.json" ]; then
  cp "$OLD/cohort/context_scaler.json" "$NEW/cohort/context_scaler.json"
fi
```

The runner checks the training-manifest hashes; copying these files does not copy labels or outcomes from 2013 into fitting. The `train` command also works without any copied files—it will rebuild the full pre-2013 feature set.

## Local commands

```bash
git fetch origin
git switch full-2013-audited
git pull --ff-only
python3.14 -m unittest discover -s pilot -p 'test_*.py' -v

# Train under the correctly enforced inverse link, using a fresh default ROOT.
caffeinate -i bash pilot/run_final_2013.sh train

# Optional lightweight VF-GAP comparison (one outer, 25 inner, 60s/oracle).
caffeinate -i bash pilot/run_final_2013.sh gap

# Optional read-only audit: reads existing solver certificates, no new MIPs.
bash pilot/run_final_2013.sh audit

# Whenever you want to evaluate the currently fitted models on 2013:
caffeinate -i bash pilot/run_final_2013.sh test
```

No mandatory audit acknowledgment or freeze step. The test runner includes whichever completed or partially fitted methods have usable coefficient vectors (it marks unfinished methods explicitly). The remaining prerequisite is a fitted cohort CASE policy for paired comparisons. Per-method test schedules still require a certified exact planner solution; a time-limit incumbent is not scored as if optimal. Incomplete planner solves receive a targeted retry.

GAP uses `cohort/gap_quick/` unless the earlier `cohort/gap/` run had already **completed**. The original interrupted GAP checkpoints are not deleted or overwritten. If GAP cannot certify a meaningful update within the smaller budget, its checkpoint records that; no other method is held back.

The audit report `train_audit/audit_summary.json` reads **existing** ANCHORED training certificates to describe congestion, decision changes, and training-solver status. Missing evidence is identified explicitly rather than launching dozens of feasibility or perturbation MIPs. A separate exhaustive operational investigation can be requested later.

Each evaluation automatically writes to `test/run_<hash>/test_scores.json` and `test/run_<hash>/test_summary.json`, keyed by the fitted policy and code hashes. Running `test` again with unchanged policies resumes prior work; changing policies creates a new run folder automatically. No test-set permissions, one-time seal, or freeze command are imposed.