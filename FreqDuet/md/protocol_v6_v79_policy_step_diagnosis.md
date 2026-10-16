# V79: diagnose the V78 policy-step failure

V78 improves passenger time and readiness but fails its unchanged regularity
gate in both deployment modes. Seed 933013's stochastic CV-squared effect is
positive with its interval excluding zero, opposite to the fitted direction.
Do not reduce step size, retune the coefficients, or select a seed after this.

Read only existing V78 arrays/pairs on node001-node006. No new rollout, critic,
actor/dual update, policy admission or changed gate. Keep the original V78
candidate fixed and classify its validation bank as consumed development data.

For each original seed, reproduce the fitted gradient and compute delete-one-
day jackknife uncertainty, recomputing all other-day baselines after deletion.
Report the fitted direction and individual feature gradients. These intervals
are descriptive conditional on the selected direction, not selection-corrected.
Fit the same optimizer on the first/last 16 training days separately and score
each direction on the other half, with baselines recomputed within that half.
Those fits are diagnostic only and cannot become deployed candidates.

Compute summed conditional action KL on each unchanged training trajectory
and its complete-path likelihood ratio. Report path-weight effective days as
an overlap diagnostic; do not importance-weight outcomes. State-averaged KL
does not by itself establish a small complete-day policy change.

Use the consumed paired rollouts to separate waiting/onboard exposure, report
absolute levels and action-bin shifts, and check demand-day concentration.
Do not interpret repeats or the two known actors as new training-seed evidence.

Next decision: if the constraint direction is unresolved or fails to transfer
between independent halves, repair its estimation before control learning.
If it transfers but the finite closed-loop effect reverses, test the local
approximation separately. This audit cannot establish which mechanism dominates
merely from the sign disagreement, nor justify a new long-training sweep.
