# Stage130: Current-Policy Native Credit

Stage129 confirms compact upper gains at both periods, but period100's corrected
CI [0.168861,0.440607] remains below the0.5 practical threshold. Five of six
period100 training fits hit scale1 with positive fitted endpoint slope.

Return to development roots410011/410023 and frozen Stage128 compact weights.
Collect four fresh scenes/two full noise trajectories at every upper decision
under that learned policy, not forecast. Refit the unchanged26-dimensional
compact native direction. Compare a second update against continuation of the
old parameter direction, with both updates measured at the same current-policy
Fisher radius0.000555556. Same16 calibration scenes/two panels and training-only
quadratic step [0,1]. Keep the single-step, flat, forecast, opposite-increment
and blinded controls on32 new evaluation scenes; both periods50/100 remain.
The zero-intervention and calibration baseline is the learned single-step policy.

New cost/root:4,896label+320calibration+192crossfit+448evaluation=5,856episodes/
7,027,200steps, plus explicitly reported inherited Stage128 cost.16workers+parent,
12GiB, dynamic node001-node006. Four final upper weights/root stay server-side.

Advance only with material refresh-minus-forecast gains and positive refresh
increments; refresh-minus-stale separately tests the value of relabeling.
No new confirmation roots until this development result warrants it.

## Run Receipt

Implementation `cb9bc1d6bc`, registration `4873d4b5f4`; eight tests passed.
All four learned source uppers were read on node004 with matching source fits.
Run `pointmaze_compact_continuation_stage130_pilot_20261008_r1`:
`t136504` root410011 and `t136505` root410023 accepted. Full task specifications,
fresh seed roles, incremental budget and scheduler receipt are saved with the run.

## Completed Result

Both tasks completed;11,712new episodes/14,054,400steps,46,396bytes fetched.
All four refreshed candidates select scale1 and improve over their single-step
source on new evaluation scenes. Equal-root effects:

| Period | Refresh minus forecast | Refresh minus single | Refresh minus stale |
|---|---|---|---|
|50|+1.395884|+0.478197|-0.004481|
|100|+0.917537|+0.465657|+0.249798|

The0.5 material target is met at both periods in development. Relabeling is
not universally better than a stale second step: period50 is mixed, including
root410011's-0.029643. Period100 is positive in both roots(+0.212895/+0.286701),
but root410023's calibration favored stale while new evaluation favored refresh.
Keep this transfer boundary and seek frozen six-root replication before claims.

## Limitations

This adds a policy-improvement step, not equal total distance from zero. Only
per-update Fisher radius, calibration budget and0.05 control authority match.
This is not joint-HRL, promotion, frequency specificity or independent evidence.
