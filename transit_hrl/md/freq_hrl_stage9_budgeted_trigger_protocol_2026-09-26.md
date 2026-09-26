# Stage-9 Budgeted Plan Trigger Protocol

Date: 2026-09-26

Experiment protocol: `pointmaze_budgeted_trigger_stage9_v1_development`

Frozen algorithm revision: `91b3e6919bcffe3a75d4942b0df31ae8478d69f6`

## Question

Stage 8C qualified a causal predictor of local `keep` versus `renew` value.
Stage 9 tests whether using that prediction to time upper calls improves
closed-loop PointMaze control at the same number of calls as fixed periodic
planning. Stage-8C evaluation rows are retired from evidence for this test.

## Controller And Trigger

The goal-conditioned upper/lower PPO controller trains on a deterministic
random-offset schedule: exactly one upper call in every 50-step bin, with an
offset of 0-25 steps. Its SMDP upper transitions record actual 25-75-step
option durations, and lower intrinsic GAE ends at the actual waypoint change.
The external task return, not lower intrinsic reward, is the trigger endpoint.

On independent branch-fit paths, exact-prefix `keep`/`renew` pairs use the
same random-offset prefix. The 39-feature causal interaction predictor and a
170-feature current-only quadratic predictor each choose ridge alpha by
leave-one-path-seed-out fit MSE. Their deployment thresholds are the 75th
percentiles of fit-path out-of-fold predictions; held-out episodes never tune
them. Event labels and privileged regime state are excluded from both models.

Within each 50-step bin, a trigger checks the current score at offsets
0, 5, 10, 15, 20, and 25. It calls the upper planner at the first threshold
crossing or at the offset-25 deadline. The first bin always calls at step 0.
No predictor check invokes the upper planner. Every mode makes exactly one
upper call per bin and holds the new waypoint until its next actual call.
The branch-fit label uses a common 50-step continuation; the closed-loop test
determines whether that local proxy remains useful under 25-75-step options.

## Frozen Comparisons

The four schedules share each frozen trained controller and each held-out
episode path: fixed at bin start, seed-determined random offset, current-only
trigger, and causal-history trigger. The candidate is the causal-history
trigger. The primary endpoint is paired fixed minus candidate episode tracking
ISE. Secondary registered checks require candidate ISE lower than random
offset and current-only trigger, candidate episode return higher than fixed,
and the variable-duration controller to improve ISE over its paired untrained
initialization. Each comparison uses the root mean and a two-sided 95%
Student-t interval; every required lower bound must be strictly positive.

Formal optimizer roots are `208011, 208023, 208037, 208049, 208061, 208073,
208089, 208101`. Each has 8 controller-training, 8 checkpoint-selection,
8 branch-fit, and 16 trigger-evaluation paths. The root is the statistical
unit. Sequential root extension is forbidden.

Preflight root `208001` uses 1 training, 1 selection, 2 branch-fit, and 2
trigger-evaluation paths, 2 controller iterations, a 300-step horizon, and
one branch opportunity per class. It validates software and accounting only.

Each cell is a dynamically placed one-core CPU task on `node001`-`node006`.
Only compact `result.json` artifacts are synchronized; checkpoints are
disabled. Passing this development gate would motivate a separate frozen
confirmation and transfer study, not establish a domain-general result.
