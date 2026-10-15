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
time. No future labels, across-seed pooling or gate tuning.

Keep V42/V43 engineering thresholds: 30-min holding MAE <= historical, absolute
bias <=5%, passenger MAE <=105% of historical, holding coverage >=80%. Report
all origins and the common clocks separately, warmup/mature counts and widths.
Use scheduler on node001-node006 for tests, full-day replays and aggregation;
retrieve only small summaries. Validate exact archived V35 outcomes first.

## Limitations And Next Step
Calibration of cumulative endpoints does not fix state propagation or establish
correct counterfactual action response. No main-policy switch. Diagnose and fix
supported timing-contract defects first; then evaluate frozen calibration on
new demand contexts before considering any deployment change.
