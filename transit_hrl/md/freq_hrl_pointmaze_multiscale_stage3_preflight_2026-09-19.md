# PointMaze Multiscale Stage-3 Preflight

Date: 2026-09-19

## Outcome

The eight-cell preflight
`pointmaze_multiscale_stage3_v1_preflight_20260919_r1` completed successfully as
scheduler tasks `t95327` through `t95334` on node004, node005, and node006.
Every task recorded exit code zero and the explicit
`complete: result.json written` marker. The inspected logs contained no
traceback, exception, kill, or segmentation-fault marker.

All eight `result.json` files were 17-29 KB and passed the registered structural
checks:

- protocol and runtime versions matched the frozen registration;
- each rollout had exactly 64 steps, no environment termination, and one final
  time-limit truncation;
- each HRL rollout had three upper decisions and three lower option boundaries;
- every row had `protocol_valid=1`;
- parameter-budget ratios ranged from 0.9973 to 1.0051;
- clean measurement, slow-action, and fast-action stress RMS were exactly zero;
- mixed-stress RMS values were nonzero and exactly paired across all four
  methods for the shared episode seed;
- every policy produced a finite training update.

The preflight used two training updates, one optimizer root, and one held-out
episode per cell. Its success, return, distance, and factorial intervals are
therefore software diagnostics only. The preflight does not support or
contradict a performance claim. It authorizes the frozen 64-cell development
matrix without changing stress amplitudes, model settings, endpoints, or claim
gates.
