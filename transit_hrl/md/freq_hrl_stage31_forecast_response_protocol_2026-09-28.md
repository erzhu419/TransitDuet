# Stage-31 Forecast-to-Plan Response

Freeze roots 209011/209061 (preflight 208001), Stage-26 controller weights,
Stage-28 training pairs and Stage-30 motion weights. Controller/physical/motion
updates and reconstruction are zero. Factual replay tolerance remains 1e-8.
Use only complete observed training prefixes: 310 pairs per root, two preflight.

Response features: current23, candidate-minus-position2, candidate-minus-old2,
two current plan-target squared distances, forecast displacement8, old/new
plan-target loss changes4+4, velocity-displacement alignment4. The 49-column
design uses current causal upper proposals and frozen 10/25/50/100-step target
forecasts. Five equal-width forecast views: history, current, shuffled, lag-one,
zero. Raw history/current 31-column heads are additional controls. Unit ridge,
training-only scales (constant threshold 1e-8), unpenalized intercept and
10/25/50/100/150-step response rates are fixed; no model/window/threshold search.

Fresh query bases 3309000/3310000/3311000, offsets 101-108 (101-102 preflight).
Per path select 15 distinct bins from 2 onward, seed [root,path,31031], with
offsets 0/5/10/15/20 three times; preflight has one check per path. Reuse true
100-step keep-old-plan and 150-step equal-call settlement with full lower
feedback. Each arm executes one post-check upper call, at check or check+100.
Candidate proposal computation is shared across all methods and counted,
not claimed free or evidence of reduced planning inference cost.

Prediction gate: history settled-rate MSE below zero-value and every learned
control. Decision gate: paired settled ISE benefit strictly positive against
all six learned controls and always-keep/renew. Both gates must pass on both
roots; ties fail. Path-block percentile CI uses 4096 frozen resamples, seed
[root,31039], and is descriptive, not an alternate gate or selection rule.

Full 209011/209061: 171800/173700 new paired steps plus 1200 factual each =
347900 total; 620 training/240 fresh pairs, 860 candidate-proposal calls,
14 critic fits/multi-RHS solves (70 RHS). Preflight: 1030 paired+300 factual,
two fit/two query pairs, four proposals, seven fits/35 RHS. Raw arrays and
weights stay remote; sync result JSON only. Dynamic scheduler node001-node006,
16 workers/17 CPU/24 GB, preflight one worker/2 CPU/3 GB. Twenty-four tests pass.

## Limitations

This is policy-specific paired local decision validation under reused roots,
not episode deployment or reward improvement. Candidate inference has a cost;
gross early curves have unequal executed calls. Stage-9 remains the performance
reference; earlier failed gates and the negative preflight are preserved.
