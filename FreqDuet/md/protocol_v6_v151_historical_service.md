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

Execution source1b783ea8f3. Preflight t199152 passed12 tests and consumed baseline
reconstruction, then stopped before a new rollout because the stage-local
runtime_optimization_20261014 link was missing. Linked the established shared
proof directory, without changing code, then t199164 passed both calibration
smoke paths(exit0). Calibration t199175--t199180(8 workers/shard), seal t199181,
confirmation t199182--t199187(8 workers/shard), aggregation t199188. Dependencies
use only small synced receipts. No scientific confirmation result at submission.

## Confirmation Readout

All14 formal tasks exited0, no retries. Complete192 actor/day pairs,384
confirmation paths; the preregistered **joint prediction gate passed**.
Full same-follower target-ready ETA MAE(seconds), on the same16 fresh days:

| Mode | V141 | V149 | V151 joint | Joint minus V149,95% CI |
| --- | ---: | ---: | ---: | --- |
| old,clearance | 38.927 | 36.956 | 34.731 | -2.225 [-2.896,-1.567] |
| handoff,clearance | 38.927 | 36.938 | 34.741 | -2.196 [-2.895,-1.505] |

All-day MAE improves0.469/0.369s(old/handoff),hour13 improves2.315/2.327s.
Matched-identity source-switch MAE burden falls0.35037->0.16957s per valid
matched decision; paired delta -0.18080,CI[-0.23915,-0.12971]. Clearance MAE
improves for all12 actor margins and15/16 day margins;2752010 is the exception.

Read-only audit t199238/source76f0bf7946 exited0,4 tests passed, all original
statistics and point-shift budgets reproduced. No new rollout, refit or update.
The all-day downward point shift is2.145/2.022s: downstream-service changes
contribute1.749/1.748s,current residence only0.045/0.044s. Active recovered-tail
queries improve conditional MAE86.82->45.79s /88.45->45.47s. These are fixed-path
diagnostics, not passenger-control gains or true component-error attribution.

### Limitations And Next

All-day signed bias becomes about-2.68s; mean absolute cell bias increases
2.419/2.320s despite lower MAE. Exploratory hour2 MAE worsens0.786/0.877s;
handoff hour10 also worsens0.380s. Four handoff clearance-launch queries worsen
conditional MAE0.036->40.955s(no crossed CI for this sparse subgroup).
V146 is unchanged and does not establish full-ETA interval coverage here.

Keep V151 sealed as a positively validated forecast reference, not a controller
replacement. The actual downstream-service and running/residence comparison
was completed in V152(t199247,source9fbd08c51b): service MAE
improves overall;hour2 combines service degradation and lost cancellation in the
point estimates;the four handoff launches improve service but expose existing
running underprediction. Next localize running errors by segment/time, without
hour-specific exemptions, global offsets or fitting on these confirmation days.
Online causal equivalence is still required before control use.

Local small summaries: results_freqduet/protocol_v6_v151_historical_service/
workflow_1b783ea8f3/summary/status.json and readout_76f0bf7946/status.json.
