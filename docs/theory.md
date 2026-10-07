# Mathematical implementation notes

## Planner and cost

An instance t is one room group on one day. For surgeon-day s and eligible room r, x_sr is binary and sums to one over eligible rooms. Binary y_r is one exactly when at least one surgeon-day uses r. With n_r assigned cases,

\[
L_r(d)=\sum_{i\in r}d_i+30(n_r-y_r),\qquad B_r=480y_r.
\]

An unused room has zero load, capacity, and cost. The cost is

\[
J(z;d)=\sum_r\{[480y_r-L_r(d)]^++1.75[L_r(d)-480y_r]^+\}.
\]

Since every case is assigned, total load is sum_i d_i + 30(n-R), where R=sum_r y_r. Thus

\[
J(z;d)=510R+2.75\,OT-\sum_i d_i-30n.
\]

This is an algebraic equivalence, not an additional opening charge. An extra occupied room pays off precisely when its reduction in total overtime exceeds 510/2.75, approximately 185.45 minutes. A room merely exceeding that overtime level is not sufficient if indivisible surgeon-days or eligibility prevent the required reduction.

Release, nonempty candidate sets, and eligibility fallback make the feasible assignment set nonempty. It is finite and independent of durations. Fixing a schedule fixes y and the turnover term. Therefore Assumption 3 retains its cost form with capacity 480y_r and turnover 30(n_r-y_r).

The ratio 1.75 is a modeling input supported by Dexter and Macario (2004), not an estimate of UHN staffing costs. Room-opening formulations, including Wang et al. (2019, 2024) and Deng, Shen, and Denton (2019), provide related modeling context; operational decision timing and release feasibility remain application assumptions.

## Response and reachable oracle

For displayed correction u, adoption alpha, radius h, and skepticism intensity one,

\[
F(u)=\operatorname{sgn}(u)\min\{\alpha|u|,[h-(1-\alpha)|u|]^+\}.
\]

Clip the display to [max(-180,1-b_i),180] before applying F. Its continuous image is the correction interval

\[
[-\alpha\min(h,180,b_i-1),\;\alpha\min(h,180)].
\]

VF-Direct trains with F(u)=u, so it does not require Assumption 2's finite-rejection condition. Its test recommendations nevertheless pass through the selected scenario response.

Let D_t be the Cartesian product of the reachable duration intervals. The response-limited oracle is

\[
O_t=\min_{d\in D_t}\ \min_{z\in\arg\min_{z'\in Z_t}J_t(z';d)}J_t(z;a_t).
\]

Ties favor the oracle. Every evaluated policy supplies a reachable d and a primary-optimal schedule; consequently O_t is a floor for its realized cost, even though the policy planner has an outcome-blind deterministic tie rule.

The implementation sums case duration intervals within each surgeon-day. Every point in that summed interval can be disaggregated into casewise reachable durations by proportional interpolation between the endpoints. Binary/continuous products attach these totals to selected rooms.

The master minimizes realized cost subject to its current competing-plan constraints. For each stored plan z', impose J(z;d) <= J(z';d). The common term -sum_i d_i-30n cancels. Each competing plan's positive-part overtime is represented by an exact maximum graph; an unconstrained epigraph on this right-hand side would invalidate the oracle. The selected plan's overtime can use an epigraph because the inequality requires it to be no larger than each competing cost.

After a master solve, solve the daily follower at the induced durations. A cheaper feasible follower plan supplies another valid constraint. Primary-optimal follower plans supply feasible upper bounds; an optimistic master assignment supplies one only after its follower optimality is confirmed. Booked gives the first upper bound. The realized-duration oracle and master relaxation dual bounds supply lower bounds.

An exact response-limited result requires both a globally optimal master and a certified follower check, with matching bounds within the stated tolerances. Otherwise preserve its bracket and cuts. Never promote a timeout or a cached unfinished result to exact. Feasible upper bounds include an assignment and reachable-duration witness. Summing daily lower and upper bounds gives a valid aggregate bracket.

## Envelope and training certificate

The per-minute envelope is

\[
\theta(v)=1.75[v]^++[-v]^+,\qquad \Theta_t(v)=\sum_i\theta(v_i).
\]

The number 2.75 is the sum of the two slopes; it is not the envelope itself. For a fixed assignment, each room's cost change is bounded by theta of its aggregate duration change. Subadditivity then gives Lemma 1 unchanged:

\[
J_t(z;d')\le J_t(z;d)+\Theta_t(d'-d).
\]

For a feasible schedule library L_t, let Q_t^L(d)=min_{z in L_t} J_t(z;d). If z(d) is planner-optimal,

\[
J_t(z(d);a_t)\le Q_t(d)+\Theta_t(a_t-d)
                 \le Q_t^L(d)+\Theta_t(a_t-d).
\]

This proves the finite-library part of Proposition 4 for the daily model. Enlarging a library can only tighten the bound. If the library contains a realized-duration optimum, applying the envelope a second time gives the terminal bound 2.75 times the casewise absolute error. With a lower bound q_t on the realized oracle, use

\[
\Gamma_g(w)=\frac1{D_g}\sum_{t\in g}
 [Q_t^L(d_t(w))+\Theta_t(a_t-d_t(w))-q_t].
\]

Group objectives average over their training days D_g. Certificates for the complete study sum the daily contributions; a group mean must be multiplied by D_g before summing. Add lambda_g times the L1 norm of nonintercept coefficients for the training objective. The final learned and zero policies are evaluated on identical final libraries and q_t values. These certificates concern training regret, not a guarantee of out-of-sample improvement.

Because every policy-facing daily planner has a certified optimum and the fixed deterministic tie rule, Remark 2 requires no planner-suboptimality correction in completed results. Pending daily solves cannot be used to claim this conclusion.

## Fixed-plan convex updates

For alpha<1, q=1-alpha and T=h/q, write F(u)=P(u)-N(u), where

\[
P(u)=[u+h]^++q[u-T]^+,\qquad
N(u)=q[u+T]^++[u-h]^+.
\]

For a fixed schedule and its occupied rooms, set e_i=a_i-b_i and kappa_r=sum_{i in r} b_i+30(n_r-1)-480. Let P_r and N_r sum casewise P and N. Define

\[
A_i=\max(e_i+N_i,P_i),\quad
O_r=\max(\kappa_r+P_r,N_r),\quad
I_r=\max(N_r-\kappa_r,P_r).
\]

Proposition 5 carries over: the fixed-plan majorizer equals G-H, with

\[
G=\frac1{D_g}\left[-\sum_i e_i+2.75\sum_i A_i
                       +\sum_r(1.75O_r+I_r)\right]+\lambda_g\|w_{-0}\|_1,
\]
\[
H=\frac{2.75}{D_g}\sum_i(P_i+N_i).
\]

Every case appears in exactly one occupied room, so H does not depend on the selected schedule. Released rooms have no contribution. For Case-Error alone, omit room terms and use H=(1.75 sum_i P_i+sum_i N_i)/D_g. For a direct response, P(u)=u and N(u)=0 are affine and the fixed-plan subproblem is convex.

The code represents hinges and maxima with linear epigraphs, linearizes H at the current weights, and adds a quadratic proximal term for nonlinear responses. It checks the actual fixed-plan objective before accepting an update. No response-layer binary variables or case-level assignment variables enter learning.

Corollary 2 remains a worst-case bound, not a managerial assertion that smaller display caps improve realized cost. The cap is fixed at 180 rather than selected from test performance.

## Historical scores and penalty

At date t, use only earlier dates in the same room group. For service s and a procedure or surgeon category c, the score is

\[
\frac{n_{sc}}{n_{sc}+k}\left(\bar e_{sc}-\bar e_s\right).
\]

The service/category cells may have unequal counts. With K observed cells, S services and N observations, compute pooled within-cell MS_W with N-K degrees of freedom and service-centered between-cell MS_B with K-S degrees of freedom. The effective cell size is

\[
n_0=\frac{\sum_s[N_s-\sum_c n_{sc}^2/N_s]}{K-S},\qquad
\widehat\sigma_B^2=(MS_B-MS_W)/n_0.
\]

Use k=MS_W/sigma_B^2 when defined and positive; otherwise use full shrinkage to zero. An unseen cell scores zero. The ANOVA moments themselves obey the earlier-date cutoff; final training moments are frozen for test cases.

For standardized nonintercept features, simulate 1,000 independent vectors of uniform draws and take q90 of

\[
\max_j\left|\sum_i\left(\frac7{11}-1\{U_i\le7/11\}\right)x_{ij}\right|.
\]

Set lambda1_g=2.75(1.1)q90/D_g, and lambda_g=alpha*lambda1_g. VF-Direct uses alpha=1. Case-Error and VF use the same daily loss normalization and lambda. Belloni and Chernozhukov (2011) motivates this noise calibration; its statistical guarantees are not claimed for the nonconvex VF objective.

## Sources

- Dexter and Macario (2004), *When to Release Allocated Operating Room Time to Increase Operating Room Efficiency*, Anesthesia & Analgesia 98:758–762. [Article](https://staff.washington.edu/mikeaa/SIP3/SIP3BestPractices/When_to_release_OR_time.pdf).
- Belloni and Chernozhukov (2011), *L1-Penalized Quantile Regression in High-Dimensional Sparse Models*, Annals of Statistics 39:82–130. [DOI](https://doi.org/10.1214/10-AOS827).
- Deng, Shen, and Denton (2019), *Chance-Constrained Surgery Planning Under Conditions of Limited and Ambiguous Data*, INFORMS Journal on Computing 31:559–575. [DOI](https://doi.org/10.1287/ijoc.2018.0835).
