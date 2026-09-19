# PointMaze Multiscale Stage-3 Preflight

Date: 2026-09-19

## Outcome

**Superseded after design review.** The software checks below remain an
accurate record of V1 execution, but the preflight no longer authorizes a
performance run. V1 removed current physical feedback from the upper
multiscale policy, contrary to the adopted GPT6 design. All 64 subsequently
started development tasks (`t95604`-`t95667`) were cancelled before completion.
No V1 partial output may be used as performance evidence.

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
contradict a performance claim. Its former authorization of the V1 development
matrix is withdrawn. V2 requires a fresh preflight and fresh optimizer,
training, selection, and evaluation seeds.
