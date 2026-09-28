# Stage-29 State-Response Result

Frozen implementation `d5f4bfe3d9`; tasks `t101734/101735` completed on
node006/node005. All four models used the registered 64 epochs, with no
outcome-driven setting changes. The negative preflight remains archived.

| Root | History/action physical MSE | History/blind physical MSE | History/action effect MSE | Zero-effect MSE | History/action target-rate MSE | Current/action target-rate MSE |
| --- | --- | --- | --- | --- | --- | --- |
| 209011 | 0.000808114 | 0.320481001 | 0.000522313 | 0.542958177 | 0.485247385 | 0.481691133 |
| 209061 | 0.000801580 | 0.332141748 | 0.000411991 | 0.519579002 | 0.561050101 | 0.553124607 |

Physical and effect errors use training-only delta scales. Both action models
pass the action-response gate on both roots, beating their matched blind
controls and zero effect as required. History target-rate MSE is 0.74%/1.43%
worse than current/action, improving on only 2/16 evaluation paths. Both
history gates and therefore the joint development gate fail.

Totals: 419040 new environment steps, 7296 fit/3648 query transitions,
320 effect pairs, eight fits/14848 optimizer steps, zero controller updates
or reconstruction. Server-side independent recomputation matched all reported
errors, effects, forward tapes, scales and counts; overlapping windows checked
3632/3648 next-state/action labels. Only 664221 bytes of result JSON were
retrieved; raw arrays and model weights remain on the computing nodes.

## Next Step

Retain the qualified action-response component. Separate exogenous-motion
inference from physical-response training, exposing the causal velocity
information demonstrated in Stage-27 before another plan-value learner.
Shared-latent interference is a hypothesis, not an established failure cause.

## Limitations

Two reused controller roots provide development evidence. Action qualification
is against action-blind/zero-effect controls, not a policy or reward improvement.
History/action physical-coordinate 95% coverage spans 86.90%-97.15%; uncertainty
is not fully calibrated. Known-action forward replay uses future action tapes.
