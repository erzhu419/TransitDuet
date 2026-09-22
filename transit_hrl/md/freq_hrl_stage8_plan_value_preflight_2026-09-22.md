# Stage-8 Plan-Value Preflight

Date: 2026-09-22

Run: `pointmaze_plan_value_stage8_v1_preflight_20260922_r1`

Tasks: `t100396`, `t100397`

## Outcome

The two registered cells completed on `node004` and `node006`. Only their
compact `result.json` files were synchronized. The preflight passed and
authorizes the unchanged 16-cell development matrix. It is not performance
evidence and does not authorize Stage 9.

## Audit

- Both methods used runtime Gymnasium 1.2.0, Gymnasium-Robotics 1.4.2,
  MuJoCo 3.2.7, SciPy 1.13.1, and PyTorch 2.5.1+cu121.
- The observable task history contained 64 samples by 6 channels. Base upper
  and lower states were 390-dimensional; the privileged upper state was
  394-dimensional.
- The history model had 267,018 trainable parameters. The oracle model had
  268,042, a registered capacity ratio of 1.0038.
- Each cell retained three compact history entries, selected iteration 1, and
  completed 32 finite optimizer updates.
- Each cell produced one paired untrained row and exactly seven diagnostic
  rows for the registered held-out seed.
- Fixed, perturbed, and all delayed-event schedules used exactly five upper
  calls. The stale-plan control used one. Every option-duration sequence
  summed to the exact 240-step episode length.
- Both methods used the same hidden-regime changes at steps 129 and 221 and
  the same force-pulse and distractor paths.
- The event-conditioned causal distinguishability maximum was 0.01 seconds
  with zero censoring.
- Both scheduler tasks were dynamically placed with `require_node=null`; no
  cell was pinned to its eventual node.

The analysis path also completed on the preflight artifacts. With one
optimizer root, every confidence interval is non-confirmatory and the expected
decision is `stage9_not_authorized`. This validates the stop gate; it is not a
negative performance result.

## Next Step

Run the frozen eight-root by two-method development matrix. Analyze only after
all 16 cells complete. No roots may be appended after inspecting the result.
