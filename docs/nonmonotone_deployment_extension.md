# Non-monotone deployment extension of the existing B-training grid

This is the replacement PRIMARY evaluation for the existing saturation-only
training experiment. The B-trained models, lambda settings, display choices,
and B evaluation results remain intact. The new analysis evaluates the SAME
frozen recommendations under the non-monotone surgeon response (C).

## Scientific specification

All policy training remains B (monotone saturation):
    F_B(u;alpha,h) = clip(alpha*u, -alpha*h, alpha*h)

The four existing training pairs are alpha=.5/.8 x h=30/60. Both CASE and
VF were fitted independently under each pair, with L1 lambda x1.0.
NAIVE is the single response-unaware fitted policy. BOOKED is unchanged.

New PRIMARY evaluation under C:
    F_C(u;alpha,h) = sign(u) *
      min(alpha*abs(u), max(h-(1-alpha)*abs(u),0))

C uses the same six combinations: alpha=.5/.8 x h=20/30/60.
The existing six B deployment scenarios are retained as a SECONDARY
model-checking comparison. The combined report therefore contains
12 scenarios x 10 labels (BOOKED, NAIVE, four CASE and four VF fits).

The displayed advice is constructed ONCE under each policy's TRAINING B
scenario, then frozen and passed to EVERY evaluation response. Advice
is NEVER recalculated under the deployment scenario. That matters when
h_train=60 and h_eval=20 or 30; overshooting advice is attenuated or
rejected under C even though B would simply saturate it.

Example: displayed u=50 at evaluation alpha=.8,h=20 yields B correction
16 and C correction 10. At alpha=.5,h=20, B gives 10 and C gives 0.
The one-minute minimum case duration is applied after the response.

NAIVE may display values up to 180, so its B-versus-C outcomes can differ
even when a CASE/VF policy is trained at exactly the deployment alpha/h.

C is a stylized behavioral assumption, not an empirically observed true
response: the UHN dataset contains no responses to displayed advice.

## Existing-file and cache reuse

No changes were made to the earlier runner, response_grid compiler,
CASE/VF learning routines, or training source fingerprints. Instead,
pilot/extend_response_grid_evaluation.py reuses the existing B-grid
output folder and creates response_extension/ files within it.

If original B training/evaluation is complete:
- all four CASE/VF pairs and NAIVE are reused without refitting;
- the old frozen_advice/evaluation_displays.csv is read and validated;
- all 60 previous B scenario-arm cells are reused without re-solving;
- the same original plans/content_addressed cache is used;
- ONLY new non-monotone duration vectors require planner solves;
- 60 C cells are saved individually, allowing interruption and restart.

If the prior run is incomplete, the new ONE command can finish missing
training/scenario cells with the existing checkpoint and planner cache
before beginning the C layer. NEVER run the old process concurrently
on the same artifact folder.

The original run_contract.json, selected_policies.json, frozen advice,
B cells and B-only summary.json are left unchanged when complete.
The extension owns a separate extension_contract.json, C cells, and
combined primary report, all under response_extension/.

## One command

After any currently running experiment has EXITED:

    git fetch origin
    git switch --track origin/feature/response-grid-nonmonotone-evaluation
    python -m pytest -q tests/test_response_grid_nonmonotone_extension.py

Then run the following using the SAME existing output root:

    set -o pipefail
    caffeinate -i python -u pilot/extend_response_grid_evaluation.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --cores 15 --plan-seconds 180 \
      2>&1 | tee -a artifacts/response_saturation_grid/run.log

If that branch exists locally, switch to it and git pull --ff-only rather
than using --track. Match any nondefault scientific arguments from the
original B run: contract verification deliberately rejects mismatches.

## Results

Use response_extension/summary.json as the new principal result;
the B-only summary.json is preserved only for the secondary analysis.

Other outputs:
- response_extension/evaluation_cells/NONMONO_A50_H20/... and five
  other C-response directories, one component-level file per method
- response_extension/cost_grid.csv: all 12 scenarios, each evaluated on
  its own ten-arm common-certified component sample
- response_extension/cost_grid_global_matched.csv: all 12 scenarios
  restricted to the SAME certified component sample
- response_extension/summary.json: 12 scenario-specific paired-week
  comparisons, C-B differences on a fixed common certified sample,
  modeled idle/overtime costs and MAE for every arm
- response_extension/extension_contract.json and stage_wall_times.json

The evaluation is exploratory: 2013 data have been repeatedly examined
during development. These are model-based counterfactual operating-room
costs, not observed hospital savings. A lower cost in C indicates better
performance under the assumed non-monotone response, not proof that C
describes real surgeons.
