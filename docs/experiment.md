# Experiment specification

## Cohort

Training is July 2011–December 2012; test is January–June 2013. The workbook is the sole source. Room labels are normalized by removing whitespace and uppercasing, then matched to explicit lists:

- TGH: OR1–OR17, OR19, OR21.
- TWH main: OR101–OR111, OR114, OR115.
- TWH day surgery: DS101–DS104.

A case must be noncancelled; cancellation means a nonempty cancellation reason or a recorded cancellation date. Require all four timestamps, `room entry <= surgical start < surgical stop <= room exit`, and room duration strictly between zero and 1,440 minutes. The raw booking must be positive; the duration used is raw booking plus one. The raw five-minute-minus-one grid is checked before learning.

Require a weekday that is not a closure date. A morning list is a distinct room with at least one noncancelled case entering between 07:00 inclusive and 12:00 exclusive. Combined TGH/TWH-main morning room counts are computed on the complete weekday calendar. A closure has fewer than 25% of the training-period median count; the supplied workbook yields a training median of 25 and 26 closure dates. The fixed training threshold is also used on the test calendar.

Scheduled eligibility means room entry between 07:00 inclusive and 17:00 exclusive, or a decision-to-treat calendar date at least one day before surgery. Exclude records labelled emergency. Check every pair of retained-scope, timestamp-valid cases in the same room/date; an overlap strictly greater than 15 minutes flags both cases, including nested intervals.

Only an otherwise eligible invalid record can contaminate its room-group–surgeon–date. Remove the whole such surgeon-day after finding timestamp failures and overlaps. Date identification uses the first usable timestamp in this order: room entry, surgical start, surgical stop, room exit. This fallback identifies an invalid record's contamination key; it does not repair that record. A record with no usable date is removed without inventing a date. TWH main and day surgery have separate contamination keys.

The row manifest is authoritative. Excel rows are numbered from 2, including the header offset. No count-matching search, weekly size filter, booking cap, room-prefix selection, or case rescheduling across dates is used.

| Group | Training cases | Test cases | Training days | Test days |
|---|---:|---:|---:|---:|
| TGH | 9,055 | 3,086 | 372 | 123 |
| TWH main | 11,183 | 4,158 | 372 | 123 |
| TWH day surgery | 4,268 | 1,530 | 354 | 117 |
| Total | 24,506 | 8,774 | 1,098 | 363 |

PMH is outside the main analysis because contradictory timestamps concentrated in one room and the removal of affected surgeon-days materially alter its two-room workloads. This exclusion does not establish that PMH staffing was irrevocably fixed. Cleaning also removes workload in the included groups; daily case, booking, surgeon, and room losses are exported. Historical comparisons use matched retained cases and can understate load in rooms shared with removed surgeon-days.

## Information and planning

One instance is one room group on one date. Candidate rooms are those with retained cases that day. A surgeon's cases within this group/date form one indivisible assignment unit. Cross-group surgeon-days are separate; the model does not coordinate their clock times. All open rooms are occupied, and rooms may be released in all three groups. There are no hard overtime limits, lunch breaks, opening charges, downstream resources, or day-to-day case moves. Sessions are 480 minutes with 30 minutes between consecutive cases.

A room is eligible for a surgeon-day only if every service in that unit used that room in the training cohort. If the intersection with candidate rooms is empty, all candidate rooms are eligible, and both surgeon-day and case fallback counts are reported. Release and nonempty eligibility imply assignability without needing enough surgeons to fill every candidate room.

The planner minimizes cost, then squared planned room loads, then the surgeon-to-room assignment lexicographically. Surgeon identifiers and room labels use ascending string order. Ties do not consult realized durations. Exactness means an optimal solver status at every required stage under the recorded numerical tolerances: relative MIP gap zero, absolute gap 1e-8, feasibility/integrality/optimality tolerances 1e-9, and final cost validation 1e-6. Solver tolerances are numerical tolerances, not claims of symbolic arithmetic.

When rooms have identical eligibility columns, first-use ordering removes equivalent room-label permutations while preserving the canonical optimum. Integral load units are used only when exact arithmetic on the supplied duration totals verifies the common grid; no durations are rounded. Otherwise loads remain continuous. Feasible previous/library assignments warm-start related solves. Cache identities include exact durations, case/group/day identities, eligibility, room and surgeon ordering, tie settings, solver version, and planner code.

Historical-one-room assigns each surgeon-day to an eligible room maximizing the number of its retained cases left in their recorded room, breaking ties by the same room order. This minimizes moved cases. It is a diagnostic projection, not a cost-trained competitor. Load percentiles pool occupied room-days; costs include only their retained workloads.

## Features and response

Each group has one training-fitted encoder. Inputs are booked duration; all training service indicators with the most frequent service omitted as reference; and the procedure/service and surgeon/service history scores defined in the theory notes. Frequency ties use alphabetical order. All penalized columns are standardized with training means and population standard deviations; constant columns have scale one and unidentifiable zero columns receive coefficient zero. The intercept is unpenalized.

For each training date, compute all history scores before adding that date's outcomes. This includes service means, category means, and ANOVA shrinkage estimates. Test histories and normalization are frozen after training. Unseen procedure/surgeon pairs have raw score zero. No other-category indicator is introduced; an unseen test service has all training-service indicators zero. The supplied workbook has no unseen test services. No admission, anaesthetic, case-mix, postoperative, room, calendar, or patient identifier enters the policy.

For every scenario, the displayed correction is bounded by 180 minutes and the displayed duration is at least one minute. The response has skepticism intensity one. Training follows the manuscript's convex coefficient domain: absolute coefficients at most 100 and every training display inside its bounds. Test displays are clipped before the response. The response formula and reachable interval are in the theory notes.

## Learning and checkpoints

The shared penalty is noise-calibrated with 1,000 draws, exact quantile 7/11, and recorded seed 20261007. Each group has one lambda1; Case-Error and VF use alpha times lambda1 and VF-Direct uses lambda1. Losses are divided by that group's training days. This is a calibration rule motivated by quantile regression, not a statistical guarantee for nonconvex VF.

Shift evaluates every integer implemented shift in `[-alpha*h, alpha*h]`. It displays shift/alpha, applies the duration floor and display cap, then the response. Thus a short booking may receive a smaller downward correction. Minimize summed realized training cost, breaking ties by smallest absolute shift and then positive sign. A search is complete only when every trial has exact daily plans.

The direct case-error initializer solves the asymmetric case loss with linear response. Case-Error uses proximal difference-of-convex updates on its response-aware asymmetric loss; its feasible starting point is the direct fit divided by alpha and projected into the training display domain. VF starts at Case-Error. VF-Direct starts at the direct case-error fit and is trained once per group. They are local policy fits, not certified global solutions of the nonconvex learning problem.

Each VF fit owns group-day libraries seeded by exact Booked and realized-duration plans. It adds exact plans at the initial policy, selects the cheapest stored plan per day, improves the resulting fixed-plan majorizer, and adds exact candidate-policy plans. A candidate is accepted only when the computed regularized bound does not increase. There are at most 10 convex updates per outer iteration; Case-Error uses at most 30. The proximal coefficient is 1e-4 for nonlinear responses; direct-response subproblems are convex and use zero. Inner stopping, solver status, coefficient-bound activity, and accepted updates are recorded.

VF stops after two successive outer iterations each improve the regularized training bound by less than 0.1%, or at 15 iterations. Reaching iteration 15 is explicitly recorded. Final learned and zero-policy bounds use the same final libraries and oracle lower bounds. A coefficient at its 100 bound is reported; the code never silently enlarges that scientific setting.

Every optimizer call is logged. Completed solves and policy updates are saved transactionally. An interrupted convex solve can retain a feasible improving candidate, but its fit remains pending until resumed. Incomplete daily plans cannot enter exact results. Incomplete response-oracle solves retain their valid brackets and cuts. No opportunity threshold aborts a run. The full-model license probe runs before expensive learning preparations.

## Pilots and evaluation

P1/P2 use training days at the minimum, median, 90th percentile, and maximum eligible-assignment counts within each group, with deterministic date ordering. P1 times Booked and realized-duration plans; P2 runs all four response-oracle scenarios. P3 times one full outer iteration at `(0.8, 60)` in each group, starting from its fitted Case-Error policy. Pilot policies are not test competitors. All training-day Booked/oracle and historical checks precede learning.

Test weights and shifts are frozen before evaluation. Repeating evaluation resumes the same policies and instances; a changed policy fingerprint is rejected before test solves. All five methods pass through each scenario response. Daily method results must be complete before publication summaries are emitted. No method-specific dropping of difficult days is permitted.

Report cost and savings relative to Booked, overtime, idle minutes, occupied room-days, and case-weighted post-review MAE. Report the realized-duration oracle and the response-limited oracle's lower/upper bounds. Use both bounds in opportunity ratios; an evaluated feasible method may tighten the oracle upper bound. A zero or unresolved zero denominator is reported as undefined, not zero.

For VF versus Case-Error, Shift, and VF-Direct, report mean calendar-week cost difference, percentage relative to the comparator's total cost, 95% percentile intervals from 10,000 shared week resamples, and weeks favoring VF. The same 26 calendar weeks and bootstrap indices are used across every group, method, and scenario. Aggregate costs within a resampled week before calculating percentages. Results are not averaged across scenarios, and a common direction is described only across the four modeled scenarios.

Interpret the results in order: perfect-information opportunity; response-reachable opportunity; opportunity captured; the three comparisons. The assumptions do not establish causal hospital savings, changes in surgical pace, feasibility for downstream resources, or generality beyond the stated scenarios.
