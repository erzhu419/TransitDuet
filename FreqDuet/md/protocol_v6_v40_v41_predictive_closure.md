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
