# V118: passive exact-path follower identity replay

Replay only the eight consumed V115 funded paths, with unchanged seven-bin law,
quota, actors, upper, sampler seeds and deployment state. Read the exact frozen
follower trip/bus ID at action-ready time, plus ETA, dwell proxy and calibration.
Record every vehicle's ready event, including services without lower decisions.

Require equality of all original endpoint/accounting fields, every original
decision row, all timing/launch NPZ arrays, and V117 reconstructed contexts. The
first path must pass before the other seven run in parallel. Only then analyze
same-object ready/arrival errors and next-service identity discrepancies:
`prediction-next = (prediction-same) + (same-next)`. Split same-object gap error
into travel ETA, service dwell and any calibration/gap adjustment.

Keep V116 restoration cohorts fixed. Compare predicted, same-object ready,
next-service ready and actual departure-pair signs. A non-neighbour same-object
pair is a diagnostic proxy, not actual CV2. No alternative action is executed.

Tests, replay and aggregation run through scheduler on node001--node006; only
the final small summary returns. No new days, fitting, training, quota allocation
change or main-policy change. Preserve missing/past labels explicitly.
