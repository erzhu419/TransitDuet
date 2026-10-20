# V151: historical service law and observed-event bridge

Prediction-only. Actors/critics/dual/main, V146(3s,0s),3600s current window and
.1/.1 native tails unchanged. V149 is development evidence, never fitting data.

12 frozen actors,8 new calibration days2751001--2751008,16 independent confirmation
days2752001--2752016; old/handoff paired paths,192+384 frozen-control rollouts.
The two-path smoke belongs to calibration, not an outcome-selection screen.

Prior: actor/stop/direction/published arrival-phase empirical service durations.
Each independent calibration day has one unit mass, split equally across its
available old/handoff modes and their events. Both deployment modes share that
same prior. The current strict completed same-phase window adds one empirical-day
mass; original all-past fallback is unchanged. Condition this duration mixture
on duration>=observed elapsed service age before renormalizing. No tail support
means abstention and the frozen V149 point, not extrapolation or fabricated zero.

Bridge observed launch before the bus-update loop, including nonzero progress
within the first segment. Same-clock ready requires the same physical follower's
earlier callback, established by saved insertion order. Other boundary cases
retain the reference. Historical labels and holding commands remain strictly
earlier than query; own-bus load is not used as follower APC.

Fixed arms: V141,V149,history-only,event-only,joint. Joint alone is gated: relative
to both baselines, both modes need clearance paired MAE/absolute-cell-bias CI95
upper<0,all-day/hour13 MAE upper<=0,and matched-identity source-switch MAE burden
upper<=0. Existing5000 crossed actor/day draws; incomplete support is inconclusive.
No best-arm selection. These are prediction gates, not controller admission or
coverage claims for the unchanged V146 interval reference.

All tests/rollouts/fitting/statistics via scheduler on node001--006. Preflight
tests and consumed baseline reconstruction precede the calibration smoke; all
calibration shards precede model sealing; confirmation waits for the sealed
prior; aggregation waits for every confirmation shard. Only small receipts and
summaries are returned locally; the model, traces, arrays and checkpoints stay
on server. Freeze the estimator before calibration and the prior before confirmation.
