# V40/V41: uncertainty and observable predictive closure

V40 issues one return forecast per trip and predicted-lead band (<=15 min,
15-30 min, >30 min), starting at 30 min on a five-minute observation grid.
Nominal 90% intervals use the last 128 already completed residuals in that
band, require 20 completed samples, and remain unavailable during warmup.
The order statistic is ceil(0.9*(n+1)), capped at n. Labels become usable only
strictly after the observed return. Report coverage, width and interval score,
including missing warmup intervals; dependent observations do not imply a
formal coverage guarantee. Replay the same four frozen policy/scenario days.

V41 adds a short-window fluid service predictor. APC boardings plus the change
in observed left-behind stock infer arrivals between service events. Initial
queues are extrapolated from the last observed stock; alighting fractions use
past APC counts. No passenger destinations, latent station queues or future
demand tape enter the forecast. Boarding respects capacity; terminal arrival
releases the fixed vehicle for FIFO reuse on the next tick.

Holding continuation is either historical stop means or a fixed-ridge response
surrogate fitted only to strictly earlier local actor decisions. The latter
reacts to projected headway, follower evidence, load and previous holding,
closing the route/holding/service/queue feedback. It is not an exact replica
of the original neural actor. Future published plans remain fixed inside the
predictor; actual policy evaluation continues causal upper replanning.

Compare 15/30-minute holding, waiting/onboard exposure and backlog forecasts
at 2/6/10 h across four reference days. Keep three additional dominant-context
interventions as diagnostics, not new independent scenarios. Validate unchanged
V37 outcomes, prefix timing, passenger mass and capacity before interpreting
model accuracy. No tail value is learned and no controller is promoted here.
All computation uses scheduler on node001-node006; only small summaries return.

## Outcome and next step

V40 smoke `t153486` and final V41 smoke `t153573` pass; the latter passes 15
tests. Seven formal replays (`t153578-t153582`, source `2d230fa317`) exactly
match V37. Aggregations `t153583/t153584` and operational audit `t153585`
(analysis `4a25148f3f`) retain all four reference days and three diagnostic arms.

Across 1044/1028/953 short/medium/long forecasts, 952/913/830 have calibrated
intervals. Coverage is 88.2/89.2/91.2%, with mean widths 292/531/657 s.
This describes uncertainty but does not resolve actions separated by 15-45 s.
The intervals calibrate the pooled ETA model, not the fluid response model.

At 15 min, response vs historical holding predicts matched local actions with
MAE 10.79 vs 15.33 s; waiting/onboard exposure MAEs are 23051/18936 vs
23470/20711 person-seconds. Yet total holding MAE worsens from 213 to 367 s;
at 30 min it worsens from 337 to 675 s, and onboard exposure also worsens.
Thirty-minute predicted-return MAE improves from 122.63 to 103.42 s (predicted
returns only; window accuracy 85.7 to 90.7%). The response surrogate is not
accepted for action selection. Next replace mean-regression-plus-rounding with
a categorical holding distribution, validate trajectory accumulation on fresh
contexts, and calibrate the resulting return trajectories before learning tail
value. No gate tuning or controller promotion on this development cohort.
