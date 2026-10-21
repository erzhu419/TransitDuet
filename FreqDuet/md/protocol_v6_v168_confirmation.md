# V168: independent confirmation of frozen V167 maps

Source `8466f300ab`, map `e6fc851ec1`, protocol `4a6d783667`; 17 tests passed.
High/manual tasks: preparation `t208543`, 48 completed shards `t209824`-`t209871`
(zero retries), aggregation `t209872` on node006, read-only marginals `t209874`.
Reserved days2852001-2852032 / samplers1600000032-1600000063: 768 paths,
3,993,600 queries; complete actor/day/scope grids, no finite unresolved labels or
missing matched references. Bounds precede labels. Bootstrap: 5,000 draws, seed20260803.
Whole-day/active gates pass in both modes; all finite-fraction bounds pass.
Whole-day old/handoff coverage0.826554/0.826411; paired interval-score delta
-2.743 [-3.027,-2.454] / -2.735 [-3.022,-2.450] seconds (95% crossed CI).
The full joint gate is **not admitted**:
- hour2 coverage0.749483/0.749434; above-upper rate0.220570/0.220672.
- clearance above-upper upper95=0.104256/0.105779; score-delta upper95=+0.161349/+0.258261.
- hour13 above-upper upper95=0.104971/0.105569, despite better mean score.
Hour2 fails descriptively in all 12 actor marginals and 29/32 day coverage marginals;
all 32 day upper-tail rates exceed0.1. Whole-day marginal gains hold across all actors/days.
Matched coarse also undercovers hour2 (~0.749), but passes hour13 upper-tail bounds;
shared hour2 failure and conditional hour13 tail risk require distinct diagnosis.
Only 41,058-byte JSON and a small log were fetched; frozen items unchanged, no refit.
Next: read-only hour2 R/S/residence residuals and calibration/confirmation comparison,
plus hour13 upper-bound contraction versus coarse. No tuning or re-admission on these 32 days.
