# VF-RAY: Direct-cost calibration along the existing VF direction

This is one additional *TRAINED* policy family built on the four existing
B-trained VF policies. It does NOT replace or rerun CASE, VF-pDCA, NAIVE,
the six original B evaluation scenarios, or the six C evaluation scenarios.

## Mathematical definition

For every training response scenario pi in
(alpha,h) = (.5,30),(.5,60),(.8,30),(.8,60), the previous solver has
already determined:
- CASE raw score a_i = X_i w_CASE;
- standardized CASE-draft five-dimensional feature H_i;
- VF increment vector theta_VF, fitted with MM+pDCA and unchanged lambda.

The ray is theta(s) = s * theta_VF. For each s, the B-implemented duration
is d_i(s) = max(1, booked_i + clip(alpha*(a_i+s H_i theta_VF),
                              -alpha*h,+alpha*h)).
We recompute the deployed HASH-SEED-0 weekly planner on d(s), using
the same physical roster, surgeon-day constraints, eligibility, turnover
and cost as the prior experiments.

The new *TRAINING* objective is strictly the original realized OR cost:

    F_train(s) = (1/number_training_components) *
        sum_week J_week(z_hash_0(d_week(s)), actual_week)

NO VF surrogate term, prediction error term, L1 penalty, or 2013
evaluation outcome is used when selecting s. A component's planner
assignment is constant over intervals of s, so this unregularized
objective is piecewise constant in ideal exact arithmetic. Its jumps
can occur at unknown, arbitrarily close locations. Enumerating them
exactly is computationally difficult.

## IMPORTANT: what is, and is not, certified

The implementation evaluates ALL selected grid points on ALL historical
TRAINING components (not an unlabelled random screen). The grid is
predeclared and uniform between 0 and min(requested s_max,64/max|theta|),
plus s=0 and s=1 inserted exactly. This retains the original VF box.
At every point the planner uses its real hash-0 tie rule, and status
'complete' is checked for every training component.

Select a fully certified point with minimum total realized training
cost; ties use the SMALLEST s. The program saves every exact planner
status and mean training direct cost to a scenario-specific ledger.

If all grid points are fully certified, the program can truthfully
claim **optimality among the finite sampled s values**. This is NOT a
global optimum along the continuous ray and NO error bound follows
solely from the grid spacing, because the cost jumps at assignment
switches. A "200-point grid is exact to one grid interval" is NOT an
objective-value guarantee.

When both s=0 and s=1 are fully certified, the selected point cannot
have greater realized TRAINING cost than either CASE or the original
VF. This is a training result only: there is no never-worse guarantee
for unseen cases or under non-monotone response mismatch.

An existing VF training-value-function upper-bound theorem may still
apply to the new policy if its assumptions are satisfied, but a prior
NUMERICAL epsilon at s=1 is not automatically a valid certificate at a
different s. The ray search does not claim to recompute such a bound.

If any s lacks a certified weekly planner solution, that point is
NOT eligible for the objective comparison. The run explicitly flags
"incomplete grid", and it does NOT claim finite-grid optimality. If
either endpoint lacks certification, it also does NOT claim the
never-worse-than-CASE/VF training guarantee.

## Cost and runtime

The default 13-point ray grid is a modest, fixed research budget, NOT
theoretical exact continuous optimization. There are four different VF
directions and 3,362 training components per direction, so in the
worst case up to about 4 * 13 * 3362 = 174,824 component-level
training hash-0 planner solves. This is materially MORE work than
13 calls to a single weekly planner. The cache eliminates identical
component/duration vectors and saves each solve immediately, but
NO 15-minute runtime claim is justified from previous runs.
Use the provided default configuration for one comparable experiment;
do not tune the number of points based on 2013 evaluation results.

## Evaluation on exactly the existing grid

After the four training-only multipliers s are frozen, the held-out
2013 recommendations are computed once under each policy's original
B training response and canonicalized for display. Each identical
display is passed through the SIX existing C and SIX existing B
deployment response scenarios; no re-fitting or re-canonicalizing
advice based on the evaluation response.

The resulting four new policy labels are:

- VF_RAY__SAT_A50_H30
- VF_RAY__SAT_A50_H60
- VF_RAY__SAT_A80_H30
- VF_RAY__SAT_A80_H60

The unchanged BOOKED, NAIVE, four CASE, and four original VF fitted
policies remain as ten comparators. Twelve scenarios times fourteen
policy labels total. Evaluation uses an explicitly common-certified
intersection across all 12 scenarios and 14 arms for its principal
cost grid, with scenario-specific paired calendar-week bootstraps
for VF-RAY versus its matched CASE, plain VF, NAIVE and BOOKED.

Under B, the same CASE and VF cost-revision maps used in ray fitting
apply. Under C, displayed advice beyond the deployment h is attenuated
or rejected. C is a stylized stress scenario, not recorded truth.

## Run once; reuse the EXISTING B/C experiment

DO NOT run any other process on the same artifact directory concurrently.
Pull the feature/vf-ray-direct-cost branch of RohamBahri/surgery_scheduling.
Keep the output directory EXACTLY the same as the previous B/C experiment.

    git fetch origin
    git switch --track origin/feature/vf-ray-direct-cost

    python -m pytest -q tests/test_vf_ray_direct_cost.py

    set -o pipefail
    caffeinate -i python -u pilot/extend_vf_ray_grid.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --cores 15 \
      --plan-seconds 180 \
      --s-max 2 \
      --grid-points 13 \
      2>&1 | tee -a artifacts/response_saturation_grid/run.log

If already checked out, use git switch feature/vf-ray-direct-cost then
git pull --ff-only instead of --track.

This one command reuses the previous completed fitted policies,
fixed training/evaluation CASE-draft feature definitions, B/C cells,
and content_addressed planner cache. If prior B/C cells are absent
it can finish them using the same original checkpoints before
calibrating rays. It never rewrites original policies, B cells,
C cells or their scientific contracts.

The new experiment owns ONLY:
- vf_ray_extension/run_contract.json, pinned to old fit and advice hashes;
- vf_ray_extension/training/<B training label>/s_###.json;
- vf_ray_extension/training/<B training label>/selection.json;
- vf_ray_extension/selected_rays.json;
- vf_ray_extension/frozen_ray_advice.csv;
- vf_ray_extension/evaluation_cells/<scenario>/VF_RAY__...json;
- vf_ray_extension/summary.json;
- vf_ray_extension/cost_grid_global_matched.csv;
- vf_ray_extension/stage_wall_times.json.

Every step is persisted. Repeating the EXACT command resumes the
remaining planner solves/outputs rather than retraining VF or redoing
the completed points. New parameter settings should use a distinct
copy of the artifact root, never overwrite old ray search ledgers.

Because 2013 has already been used for iterative model development,
its statistical intervals are exploratory. The true surgical response
has not been measured in historical data; all cost improvements are
modeled counterfactuals, not hospital savings.
