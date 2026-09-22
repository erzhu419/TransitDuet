# Stage-8B Counterfactual Plan-Validity Protocol

Date: 2026-09-22

Algorithm revision: `35c54cdee248216ce27449c40fc61f6a5db785c5`

Experiment protocol: `pointmaze_plan_validity_stage8b_v1_development`

## Why This Stage Exists

Stage 8 established that refreshing and preserving the upper waypoint matters,
but it did not qualify an event trigger. Current-regime oracle input was
inconclusive, zero-delay event calls were harmful, and nominal 250-ms calls
were better. Event relocation also exposed a fixed-duration planner to highly
variable option lengths and did not realize the named 500-ms first-response
delay.

Stage 8B therefore does not equate a detected regime boundary with a replan.
It asks the direct causal question:

> From the same current state and plan, does renewing the waypoint now reduce
> the next 0.50 seconds of integrated tracking error relative to keeping it?

No trigger is deployed in this stage.

## Paired Branch Intervention

The Stage-8 hidden-regime PointMaze task and fixed 0.50-second history HRL
controller are retained. At each registered opportunity, the deterministic
episode prefix is replayed twice from the same seed:

- `keep`: retain the active waypoint;
- `renew`: call the same upper planner once and replace the waypoint.

Both branches then run for exactly 0.50 seconds with no downstream upper call.
The lower controller remains closed loop in both branches. Prefix physical
state, task observation, 64-step history, and active waypoint must agree
elementwise before intervention. The primary label is

`renew_ise_advantage = ISE_keep - ISE_renew`.

Positive values mean renewing has local control value. Paired branch replay is
extra simulator supervision and its full primitive-step budget is reported
separately from controller training.

## Opportunity Classes

Each path contributes four non-overlapping opportunities from each class:

- regime change plus 10 ms, 100 ms, and 250 ms;
- measured short force pulse plus 10 ms;
- reward-irrelevant distractor change plus 10 ms;
- a matched control point with no nearby regime change or force-pulse start.

True event tables select diagnostic opportunities only. They are absent from
candidate features. Opportunities never coincide with the fixed upper boundary.

## Predictor Isolation

Eight branch-fit paths train fixed-alpha ridge predictors. Sixteen disjoint
paths are used once for qualification. The candidate feature vector contains
only current physical/target/waypoint state, plan age, current nuisance values,
and causal target/force/distractor history summaries. It has no future event or
regime label.

Registered comparisons are:

- plan age only;
- current plan/state without temporal summaries;
- change-magnitude baseline;
- full causal history;
- full causal history plus privileged current regime, as a ceiling only.

Top-quarter selection utility is the mean paired local advantage among
opportunities ranked highest within each held-out path. It is not a sequential
closed-loop return and cannot be reported as one.

## Gate

Optimizer root is the statistical unit. All conditions require a positive
paired two-sided 95% Student-t interval across the fixed eight roots:

1. The fixed-period history controller improves held-out ISE over its paired
   untrained initialization.
2. Renewal has positive value 250 ms after a regime change.
3. The 250-ms value exceeds the 10-ms value.
4. The 250-ms regime value exceeds force-pulse and distractor-change value.
5. Causal-history predictions have positive held-out Spearman correlation.
6. Their top-quarter selections have positive local renewal value.
7. Their selected value exceeds the current plan/state predictor.

The conjunction alone authorizes development of a budgeted Stage-9 trigger.
It does not validate a deployed trigger, closed-loop improvement, learned
belief, or frequency-specific Freq-HRL claim.

## Frozen Matrix

The preflight uses root `206001`, two PPO iterations, one seed per role, one
opportunity per class, and a 240-step episode. It is software evidence only.

The unchanged development matrix uses roots `206009`, `206021`, `206033`,
`206047`, `206063`, `206071`, `206087`, and `206099`. Each root uses 8
controller-training paths, 8 checkpoint-selection paths, 8 branch-fit paths,
and 16 branch-evaluation paths. All roles are disjoint and fresh. Sequential
root extension is forbidden.

Every single-core task may run on any of `node001`-`node006`; no task is
node-bound. Only compact `result.json` artifacts are synchronized. Checkpoints
and raw trajectories remain disabled.
