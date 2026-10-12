# Stage196: Conditional Noise Versus Critic Error

Reuse Stage195's240 states,40 development prefixes, eight conditional draws and
47 frozen fits. For each cost/fleet/nonfleet/wait head, subtract the mean-label
pair variance from observed pair MSE without clipping negative estimates.
Compare against zero action differences using the same noise subtraction.
This is conditional Monte Carlo error analysis, not a cross-seed confidence CI.

Check oracle choice transfer between odd/even four-draw halves. Split selected
false wait-feasibility into harm in both halves versus half disagreement.
Oracles are diagnostics, not deployable policies or causal performance claims.

One scheduler report task, nodes001-006 unbound. No new fits, actor updates or
native episodes. Raw draws/predictions remain remote; only summary is synced.

## Next Step
Use the observed predictor-versus-zero errors and cross-half transfer to decide
whether to change critic representation/objective or the upper action protocol.
Do not change the Stage195 joint gate or open reserved scenes from this report.
