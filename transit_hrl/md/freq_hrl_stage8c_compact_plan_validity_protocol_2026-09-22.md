# Stage-8C Compact Plan-Validity Protocol

Date: 2026-09-22

Experiment protocol: `pointmaze_compact_plan_validity_stage8c_v1_development`

Runtime protocol: `pointmaze_compact_plan_validity_stage8c_v1`

Frozen algorithm revision: `78e8493c6ba4399b3d68df4da4d353aaab03104e`

## Purpose

Stage 8B established positive delayed `renew now` value and causal-history
ranking, but its 37-feature linear predictor did not significantly improve
selected local value over the current plan/state model. Stage 9 therefore
remains blocked.

The completed Stage-8B evaluation split is now a development source only. It
was used to select one new predictor structure; none of its observations can
provide Stage-8C evidence. Stage 8C uses entirely fresh optimizer, training,
selection, branch-fit, and branch-evaluation seeds.

## Frozen Predictor Suite

All predictors target paired local ISE advantage,
`ISE(keep current waypoint) - ISE(renew now)`. They receive no event category,
event time, future value, regime label, oracle context, or distractor feature.

`current_compact_quadratic` is the strong no-history baseline. It applies all
linear, squared, and pairwise terms to 17 current physical, target-error,
waypoint-error, plan-age, target-position, and force features, producing 170
features.

`causal_dynamic_quadratic` applies the same expansion to 17 causal dynamic
features containing plan age, current errors, multi-lag target velocities, and
force magnitude. It produces 170 features and is a high-capacity diagnostic
comparison.

`causal_validity_interactions` is the registered candidate. Its 39 features
contain current errors and distances, four causal target-velocity estimates,
velocity norms, target/waypoint directional projections, short-minus-long
velocity terms, and plan-age interactions. The structure is frozen; it is not
selected again on Stage-8C results.

Each model chooses ridge alpha from
`[0.01, 0.1, 1, 10, 100, 1000, 10000]` by leave-one-branch-path-seed-out mean
MSE on branch-fit rows only. Held-out branch-evaluation rows do not affect the
transform, alpha, or weights.

## Paired Branch Contract

The controller and six opportunity classes are unchanged from Stage 8B. At
each opportunity, `keep` and `renew` replay the exact same deterministic prefix.
`keep` makes zero upper calls, `renew` makes one, both then hold the resulting
waypoint for 0.50 seconds with no downstream upper call, and the lower policy
remains closed loop. Branch transitions are extra predictor supervision and
are reported separately from controller training.

The six balanced classes are regime change at +10, +100, and +250 ms, force
pulse at +10 ms, distractor change at +10 ms, and matched neutral time. Event
tables choose diagnostic opportunities only and never enter a predictor.

## Frozen Matrix

Formal optimizer roots are `207011, 207023, 207037, 207049, 207061, 207073,
207089, 207101`. Each root uses 8 controller-training paths, 8
checkpoint-selection paths, 8 branch-fit paths, and 16 held-out
branch-evaluation paths. Every branch path contributes four opportunities in
each class. The root is the statistical unit; all intervals are two-sided 95%
Student-t intervals over the eight roots. Sequential root extension is
forbidden.

The preflight root is `207001`. It uses one training path, one selection path,
two branch-fit paths, and two branch-evaluation paths so grouped alpha selection
is exercised. It uses two controller iterations, a 300-step horizon, and one
opportunity per class. Preflight validates implementation only.

The first implementation preflight (`...preflight_20260922_r1`) exposed a
protocol-test defect: one registered branch-evaluation seed had no force-pulse
opportunity before step 240. It failed before producing evidence and its three
scheduler attempts remain recorded as two failures and one cancelled retry.
The repaired preflight changes only its horizon from 240 to 300 steps. A frozen
test now verifies all four preflight branch paths as well as every formal branch
path have complete, balanced opportunity classes. The formal 1,200-step matrix,
algorithm revision, seeds, predictors, and gate are unchanged.

## Registered Gate

All of the following root-level 95% confidence intervals must be strictly
positive:

1. controller ISE gain over its paired untrained initialization;
2. renewal value at persistent-regime +250 ms;
3. candidate Spearman rank correlation;
4. candidate selected local utility;
5. candidate selected utility minus `current_compact_quadratic`;
6. candidate Spearman minus `current_compact_quadratic`;
7. candidate Spearman minus `causal_dynamic_quadratic`.

The primary endpoint is item 5. Failure of any item leaves Stage 9 blocked.
Passing the conjunction authorizes only development of a budgeted deployed
trigger. It does not establish episode-return, closed-loop, frequency-specific,
or domain-general improvement.

## Execution And Artifacts

Each cell is a one-core CPU task dynamically placeable on `node001`-`node006`,
with no node pinning. The scheduler stages code but excludes results and data.
Checkpoint writing is disabled. Only compact `result.json` artifacts may be
synchronized locally.
