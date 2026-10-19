# Full pre-2013 training, fast audit and repeatable 2013 evaluation

This run uses **all 24,506 cleaned pre-2013 cases for fitting** (including the former 2012 validation weeks) and the separate **8,774 2013 cases for evaluation**. There is no additional validation partition. It keeps the same inverse-response assumptions (`alpha=.8`, `h=30`) and cohort features as the earlier pilot.

The established primary comparison remains **ANCHORED (ridge .001) versus cohort CASE**. BOOKED, case-only CASE, cohort LASSO, ANCHORED ridge .003 and VF are secondary. VF-GAP is **optional and exploratory**. Model settings are not changed based on past 2013 outcomes automatically; you may rerun or analyze 2013 however you wish, but repeated test-guided choices should be disclosed if presenting confirmatory inference.

## Local commands

```bash
git fetch origin
git switch full-2013-audited
git pull --ff-only
python3.14 -m unittest discover -s pilot -p 'test_*.py' -v

# Resume existing training; do not delete the artifacts folder.
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