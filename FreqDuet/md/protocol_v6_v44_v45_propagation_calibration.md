# V44/V45: segment propagation, then completed-window calibration

V44 preserves V43 predictions and the frozen U200/L200 controller. Attach
read-only arrival/ready/action/departure clocks to the real event recorder and
to eight of the 64 forecast paths at 2/6/10 h. Match by trip/stop/stage, report
missing events, 0-5/5-15/15-30 min clock errors, and travel/service/holding-pipeline
duration errors. Inspect forward/follower gaps, load and nonzero actions without
matched predecessor-departure evidence. All eight existing days are regressions.

V45 will issue 15/30 min cumulative forecasts every 15 min from 1 h. Per day,
horizon and metric, admit only windows whose end is strictly before the forecast
origin, and retain the last 16 nonoverlapping windows. Require eight completed
windows before correction; warmup stays explicitly unavailable. Mean raw
residual corrects the point forecast. Nominal 90% bands expand the bias-shifted
Monte Carlo band by the finite-sample 90% quantile of past prequential
nonconformity scores, computed using the bias that was available at each issue
time. Calibrate only holding, onboard exposure and dispatch backlog; waiting
exposure needs latent simulator queues and remains an uncorrected prediction.
No future labels, across-seed pooling or gate tuning.

Keep V42/V43 engineering thresholds: 30-min holding MAE <= historical, absolute
bias <=5%, passenger MAE <=105% of historical, holding coverage >=80%. Report
all origins and the common clocks separately, warmup/mature counts and widths.
Use scheduler on node001-node006 for tests, full-day replays and aggregation;
retrieve only small summaries. Validate exact archived V35 outcomes for the
eight regression days. Add four fresh-demand days: actors 12011/933013 x
19511021/19511027, each checked against an unobserved same-seed frozen replay.
Freeze the calibrator before examining those fresh labels.

V44: source `c1a005ca3b`, 23 tests and full-day smoke `t155545` pass; eight
exact V35 replays `t155546-t155549`, aggregate `t155557` / `7f78c88577`.
Travel MAE 36.0/35.5 s vs service 3.87/3.80 s; holding-pipeline MAE 12.6/12.9 s.
At 15-30 min, arrival-clock MAE 110/87 s, follower-gap MAE 107/90 s.
All eight days have zero nonzero actions without departure evidence; geometry
and target headways are aligned. The suspected missing-action defect is absent
on this cohort. Remaining propagation error is primarily travel timing, with
substantial downstream headway error; these diagnostics do not prove a unique
causal attribution among the coupled dynamics. Initial acquisition smokes
`t155530/t155535` failed on frozen-recorder API differences; retries cancelled.

## Limitations And Next Step
Calibration of cumulative endpoints does not fix state propagation or establish
correct counterfactual action response. No main-policy switch. Diagnose and fix
supported timing-contract defects first; then evaluate frozen calibration on
new demand contexts before considering any deployment change.
The original V45 record also contains simulator-only waiting calibration.
Operational aggregation discards that channel and chronologically recomputes
the three observable channels, checking they equal the recorded predictions.
The channels are independent; raw forecasts and actual trajectories are reused.
