# Stage-9 Budgeted Trigger Preflight

Date: 2026-09-26

Run: `pointmaze_budgeted_trigger_stage9_v1_preflight_20260926_r1`

Task: `t100579`, completed on `node004` in about 68 seconds.

## Software Decision

The one-cell preflight passes implementation and accounting checks. It
authorizes the fixed eight-root Stage-9 development matrix. The analyzer's
`stage9_development_gate_failed` value reflects unbounded confidence intervals
from a single root and is not a performance conclusion.

## Audit

- The task used the frozen Gymnasium, Gymnasium-Robotics, MuJoCo, PettingZoo,
  SciPy, and PyTorch runtime on CPU. The controller completed 32 finite PPO
  updates under the balanced-jitter training schedule.
- One training, one checkpoint-selection, two branch-fit, and two held-out
  trigger-evaluation paths were separate. Twelve branch-fit rows covered all
  six classes once per fit path, with identical paired prefixes and no
  privileged regime context in predictor inputs.
- Current-only and causal-interaction models contained 170 and 39 features.
  Ridge alpha and threshold were selected from the two fit paths only.
- Both held-out paths ran fixed, random-jitter, current-only, and candidate
  schedules. Each 300-step episode made exactly six planner calls; option
  durations were 25-75 steps. The online fixed and random-jitter rows matched
  the original controller evaluator in decision times, ISE, and return.
- The candidate checked scores without calling the planner and advanced one
  of ten noninitial upper calls in this short preflight. This is an activation
  check, not an estimate of its full-training behavior.
- Only the 245-KB `result.json` was synchronized. The local run directory has
  that result and preregistration, with no checkpoint or trajectory file.

The registered eight-root matrix is the first performance test of this
closed-loop trigger. Its roots and thresholds remain frozen.
