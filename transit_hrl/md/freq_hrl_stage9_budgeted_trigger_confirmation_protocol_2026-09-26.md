# Stage-9 Independent-Seed Confirmation Protocol

Date: 2026-09-26

Run protocol: `pointmaze_budgeted_trigger_stage9_v2_confirmation`

The Stage-9 development result is known and is not included in this
confirmation analysis. The algorithm code remains pinned to revision
`91b3e6919bcffe3a75d4942b0df31ae8478d69f6`. Controller architecture,
384-iteration training, balanced-jitter training schedule, branch-fit ridge
models, 75th-percentile threshold, 0/5/10/15/20/25-step checks, one upper
call per 50-step bin, 1200-step episode, task distribution, and all analysis
gates are unchanged from the
[development protocol](freq_hrl_stage9_budgeted_trigger_protocol_2026-09-26.md).

The frozen optimizer roots are `209011, 209023, 209037, 209049, 209061,
209073, 209089, 209101`. Each uses 8 new controller-training, 8 checkpoint-
selection, 8 branch-fit, and 16 held-out trigger-evaluation paths. No path
seed overlaps any Stage-9 development/preflight seed. The root remains the
statistical unit; no root extension or post-hoc threshold adjustment is
permitted.

The primary contrast is paired fixed minus candidate episode tracking ISE.
The four secondary gates are controller learning, candidate ISE advantage over
random-offset and current-only planning, and candidate episode return
advantage over fixed planning. All five root-level two-sided 95% Student-t
intervals must have strictly positive lower bounds. Failure of any gate is a
confirmation failure. Passing confirms the PointMaze hidden-regime result on
fresh seeds only; domain transfer remains a separate claim.

Tasks use scheduler dynamic placement across `node001`-`node006`, one CPU and
1536 MB each. Only compact `result.json` files are synchronized; no CSV or
checkpoint retrieval is required.
