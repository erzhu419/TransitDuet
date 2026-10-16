# V81: observable regularity-moment projection pilot

V80 does not identify a reliable CV constraint gradient or a finite-step
reversal. Its existing-repeat audit finds opposite centered CV signs on
16/32 demand days and unresolved between-day covariance. Do not fit another
controller to those local gradients or select a smaller V78 step.

Test a new state-local mechanism, not V78 promotion: retain both original B200
actors and their fixed V78 coefficients. For each causal observation let p0 be
B200's seven-bin law, q the V78 law, and G(a) the existing estimated two-sided
departure-regularity gain. Minimize KL(p || q) subject to sum p=1 and
E_p G >= E_p0 G, up to 1e-12 numerical tolerance. The solution is the exponential
tilt q(a) exp(w G(a)); solve its single monotone moment with Brent's method.
If q is already feasible, keep it exactly. Missing bilateral evidence gives
constant zero gains and an unchanged q. The root is state-local computation,
not an update of any production dual or a learned value.

Sample and execute this same projected distribution; deterministic deployment
uses its argmax. No post-sampling override, new action mask, action bin, state,
critic, coefficient fit or extra physical authority. Zero V78 tilt must reproduce
the exact historical B200 episode and holding trace. The moment constraint
concerns an AVL-based local estimate; it is not a guarantee for realized follower
times, deterministic modal gain, terminal CV, passenger cost or vehicle readiness.

Fresh days 1810001-1810032, both original seeds, stochastic deployment with
three event-uniform repeats and deterministic deployment once: 256 cells,
three full-day policies per cell (B200, raw V78, projected V78), 768 rollouts.
Primary is projected minus B200 in stochastic deployment; raw-minus-B200 and
projected-minus-raw are mechanism controls, not selectable alternatives.
Cluster paired bootstrap intervals by demand day after averaging repeats and
actors, matching V78. Joint gate: passenger exposure CI upper <0 and CV-squared,
readiness delay, unlaunched and incomplete CI uppers <=0. Report journey and
holding too. Deterministic deployment has the same separate gate.

Smoke on node001-node006: ten tests, exact historical zero-tilt replay and
the first complete three-policy stochastic cell, reused in the matrix. Schedule
all remaining cells in one CPU wave, then aggregate on the server. Pull only
small completion markers and summary JSON; keep all checkpoints/traces there.
No production switch, even if this two-seed pilot passes. A pass supports a
separate seed/long-run study; failure ends this fixed projection candidate,
without tuning the moment threshold or choosing another cell after the fact.

Source `4d8d5bf55b`; smoke `t166187` passed all ten tests, exact historical
B200 zero-tilt episode/holding identity and the first complete three-policy
cell. Formal shards `t166192-t166195` are assigned to node002/004/005/006
with 65/63/64/64 workers (256 total). Aggregate `t166196` waits for markers
`0-65`, `65-128`, `128-192`, `192-256`. The smoke cell is reused; no fresh
matrix effect has been used to adjust the candidate or promote the main policy.

All five formal/aggregate tasks completed, exit 0 without retries. Stochastic
projected-minus-B200: journey -0.052956 min [-0.068082,-0.037127], readiness
-23.838 s [-26.282,-21.642], CV-squared +0.000386 [-0.000381,+0.001153].
Projected-minus-raw: passenger exposure +47114 person-s [23813,72735], CV-squared
-0.001311 [-0.001799,-0.000795], readiness +18.039 s [15.981,20.312]. Projection
retains about 58% of raw passenger improvement and 57% of readiness improvement;
it repairs much of raw regularity harm but does not establish B200 non-harm.
Deterministic journey/readiness also improve against B200, but its CV contrast
and projected-minus-raw CV repair both remain unresolved. Both joint gates fail.
Projection activates on about 39% of decisions; maximum local moment deficit
is 1.02e-12. This fixed candidate ends without threshold tuning or promotion.
Next, read existing pairs on the server for waiting/onboard tradeoffs, gain
concentration and three-repeat variance; no additional rollout or candidate fit.
