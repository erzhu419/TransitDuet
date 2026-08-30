# MuJoCo v14.28 r1 engineering invalidation

Run `mujoco_v14_28_mechanism_portfolio_preflight_20260830_r1` is invalid for
scientific interpretation. Scheduler tasks `t84832`-`t84834` all exited with
code one before candidate evaluation.

The two-fold implementation constructed one paired snapshot closure from all
128 design paths, then called that closure with a 64-path fold. The paired
closed-loop guard correctly rejected the mismatched path identity with:

`ValueError: paired-relative checkpoint rows must use identical unique paths`

The repair constructs an independent snapshot closure from each fold's own
baseline rows and uses that same closure only for candidate rows from the
matching fold. A regression test now enforces the path-matched contract.

No outcome metrics were produced or read from r1. The frozen roots, Hadamard
design, actor steps, router strengths, fold split, thresholds, and validation
rule remain unchanged for r2. The failed scheduler records remain preserved;
r2 uses a new run identity and the repaired source revision.
