# Stage-8 Plan-Value Qualification Result

Date: 2026-09-22

Run: `pointmaze_plan_value_stage8_v1_development_20260922_r1`

Tasks: `t100403`-`t100418`

Decision: **`stage9_not_authorized`**

## Integrity

All 16 registered cells completed: two methods at eight optimizer roots. Four
cells ran on each of `node003`, `node004`, `node005`, and `node006`; none was
node-bound. Only the 16 compact `result.json` artifacts were synchronized.

Every cell used the registered runtime and contained 384 learned iterations,
8 training paths, 8 checkpoint-selection paths, and 16 held-out paths. The
full evaluation has 1,792 rows (`2 methods x 8 roots x 16 paths x 7
schedules`), plus 256 paired untrained rows. All rows passed the environment,
seed-pairing, finite-metric, 1,200-step horizon, planning-budget, and SMDP
duration-sum checks. The history and oracle models had 267,018 and 268,042
parameters respectively (ratio 1.0038).

## Registered Result

The primary endpoint is physical-time integrated squared tracking error
(ISE). Positive contrasts below mean lower error for the registered candidate.
Optimizer root is the statistical unit; intervals are paired two-sided 95%
Student-t intervals over eight roots.

| Registered check | ISE improvement [95% CI] | Outcome |
|---|---:|---|
| history policy learned vs untrained | 20.1592 [14.7731, 25.5452] | supported |
| fixed refresh vs stale plan | 7.5735 [6.1774, 8.9695] | supported |
| intact vs perturbed waypoint | 0.4908 [0.4140, 0.5677] | supported |
| current-regime oracle vs history, fixed schedule | 0.0255 [-0.1096, 0.1607] | inconclusive |
| zero-delay event schedule vs fixed, oracle policy | -0.7686 [-0.8890, -0.6482] | contradicted |
| zero-delay vs 250-ms-delay event schedule | -0.9638 [-1.0847, -0.8429] | contradicted |

Mean ISE was 1.5092 for history/fixed, 1.4836 for oracle/fixed, 2.2522 for
oracle/zero-delay event timing, and 1.2884 for oracle/nominal-250-ms timing.
The causal target-velocity witness detected a registered regime change after
0.0109 seconds on average, with root-level upper CI 0.0118 seconds. That
observability fact does not rescue the failed oracle-control and timing checks.

The conjunction therefore fails. No roots may be appended, and this run does
not authorize a learned belief, plan-validity critic, or event trigger as
Stage 9.

## Design Diagnosis

The negative result is not an optimizer failure: both controllers learned
large improvements over their paired initial policies, and stale/perturbed
interventions prove that the upper plan affects control. The failed premise is
more specific.

First, the supposedly hidden signed-speed regime is almost immediately
recoverable from the explicit, noise-free target-position history. The
one-hot oracle is therefore largely redundant and did not improve fixed-period
control.

Second, a regime boundary is not the same as the instant at which renewing the
current waypoint has positive value. Replanning exactly at the boundary was
worse, while waiting allowed the physical consequence to develop. A
consequence-based plan-validity decision is needed; a change detector alone is
not justified.

Third, event-call relocation changed option durations from the trained fixed
50 steps to ranges as wide as 2-112 steps (1-111 steps for nominal 100 ms).
The planner was trained only at fixed duration and did not receive the next
hold duration. Thus the timing comparison also contains a variable-duration
policy-distribution shift.

Finally, the named delay is not always the first post-event replan delay. The
realized means were 0.0000, 0.0997, 0.2426, and 0.3075 seconds for nominal
0, 0.10, 0.25, and 0.50 seconds. The nominal 0.50-second condition ranged from
0.1473 to 0.4663 seconds because retained calls could occur before the
relocated call. It cannot support a clean 500-ms latency claim.

## Exploratory Boundary

After the registered decision was fixed, a diagnostic comparison found that
the nominal-250-ms schedule improved ISE over fixed timing for both the history
policy (+0.2189 [0.1271, 0.3108]) and oracle policy (+0.1952 [0.1063,
0.2841]). Oracle versus history at that schedule remained inconclusive
(+0.0018 [-0.2057, 0.2093]). These are explicitly post-hoc diagnostics: they
cannot replace the registered zero-delay gate or authorize Stage 9.

They do identify the next defensible question. A fresh protocol should use
paired simulator branches to estimate `renew now` versus `keep current plan`
under a common downstream rule, then test whether causal state/history predicts
that value while rejecting short force pulses and irrelevant distractor
changes. It must repair actual-delay semantics and train or condition the
planner for variable option durations before any budgeted trigger is tested.

## Claim Boundary

Supported: this learned hierarchy depends strongly on refreshing and preserving
its upper waypoint.

Not supported: current regime labels improve control; immediate regime events
are optimal replan times; delay has monotone cost; a belief-triggered method is
qualified; or the result establishes a frequency-specific Freq-HRL advantage.
