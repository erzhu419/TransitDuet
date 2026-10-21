# V169: hour2 residual transfer and hour13 upper-tail tradeoff

Read-only scan `f47254d906` / `t209904`; aggregation `1b69cd3394` / `t209905`,
both high/manual on node004. Seven tests passed; 24 actor/mode buckets completed.
V168 coverage, tails, score and width deltas reproduced; V167 seal unchanged.
Only the 148,763-byte summary was fetched; no replay, training, refit or policy change.

Hour2 matched-mask means (seconds; prediction minus truth; equal episodes then actor/day):

| Cohort / mode | Running bias | Service bias | Residence bias | Joint MAE | Coverage | Upper miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration64 / old | -30.099 | -5.388 | +0.344 | 55.444 | .754947 | .215660 |
| Confirmation32 / old | -30.316 | -4.851 | -0.121 | 55.537 | .749483 | .220570 |
| Calibration64 / handoff | -30.113 | -5.386 | +0.359 | 55.434 | .755403 | .215142 |
| Confirmation32 / handoff | -30.331 | -4.855 | -0.117 | 55.518 | .749434 | .220672 |

The native-supported mask repeats this pattern: hour2 underprediction already exists in calibration, not only confirmation. Running dominates signed bias; residence still has approximately 17s MAE. All 64 calibration-day and 32 confirmation-day running-bias marginals are negative in both modes, as are all actor marginals.

Hour13 confirmation old/handoff: upper-bound delta versus coarse -2.658/-2.740s, contraction frequency .575227/.573053. Candidate-only upper misses .029547/.029941 exceed reference-only .020619/.020622: net +0.893/+0.932 percentage points. All 12 actor marginals show this ordering. Candidate-only cases contract the upper bound by conditional means 21.775/21.396s and have mean calibration support 57.17/57.48 days; this is not exclusively a low-support phenomenon. Calibration instead had slightly fewer candidate-only than reference-only misses. Confirmation joint MAE is lower (32.77/32.78 versus calibration 33.85/33.88s), so pooled MAE improvement does not establish tail reliability.

Limitations: calibration estimates are in-sample; cohorts have different demand days and are not paired by day. Residence includes holding, execution waits and inseparable current-ready remainder, not pure holding. Conditional event means explicitly omit empty event cells; miss probabilities use the full matched mask. Crossed actor/day intervals are descriptive, not new admission tests.

Next: inspect frozen calibration-key time mixtures for hour2 and the key/horizon composition of hour13 candidate-only versus repaired upper misses using existing archives. Keep V167 and all other frozen items unchanged; preserve V168 non-admission and use no confirmation-day tuning.
