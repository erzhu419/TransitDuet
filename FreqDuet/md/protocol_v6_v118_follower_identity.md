# V118: passive exact-path follower identity replay

Replay only the eight consumed V115 funded paths, with unchanged seven-bin law,
quota, actors, upper, sampler seeds and deployment state. Read the exact frozen
follower bus ID and its trip mapping at action-ready time, plus ETA and dwell proxy.
The frozen V115 runtime builds AVL snapshots synchronously from bus_all inside
prepare; freeze bus-to-trip mapping at that call, as V112 did. It has no later
development-version calibration or follower-trip attributes; do not upgrade it.
Record every vehicle's ready event, including services without lower decisions.

Require equality of all original endpoint/accounting fields, every original
decision row, all timing/launch NPZ arrays, and V117 reconstructed contexts. The
first path must pass before the other seven run in parallel. Only then analyze
same-object ready/arrival errors and next-service identity discrepancies:
`prediction-next = (prediction-same) + (same-next)`. Split same-object gap error
into travel ETA, service dwell and any gap adjustment.

Keep V116 restoration cohorts fixed. Compare predicted, same-object ready,
next-service ready and actual departure-pair signs. A non-neighbour same-object
pair is a diagnostic proxy, not actual CV2. No alternative action is executed.

Tests, replay and aggregation run through scheduler on node001--node006; only
the final small summary returns. No new days, fitting, training, quota allocation
change or main-policy change. Preserve missing/past labels explicitly.

## Outcome and decision

Source `6e138e46a7`, task `t189898` on node006: six tests, all eight paths and
41,600 decisions exact, including endpoints, accounts, timing and V117 contexts.
Only 148,503 bytes returned. Earlier observer task `t189897` was cancelled; the
runtime was unchanged. All 34,795 valid forecasts match next arrival/departure
and same-trip/same-bus ready labels; identity contribution is zero.

Cell-weighted ready bias +16.01 [13.37,17.69] s splits into ETA +15.81 and dwell
+0.193 s; pooled ready MAE is 65.65 s. Holding-sign disagreement is 36.26%
[34.41,37.85] for negative cuts, 28.89% [28.09,29.43] for quota extras. These are
fixed-time local pair signs, not changed-action returns.

Keep quota/main frozen and close this identity audit. Next check whether the
journey-average-speed ETA (34,778 forecasts) extrapolates historical holding/dwell
into remaining travel, using completed prefixes. Require improved same-object
action-sign reliability, not just MAE, before a new allocation candidate.
