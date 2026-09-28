# Stage-30 Separate Motion Result

Tasks `t101739/101740` completed on node006/node005 at `7d87dd7e8d`, with
unchanged settings and 24 tests passed. Fresh-path target displacement-rate
MSE, averaged over frozen 10/25/50/100-step horizons:

| Root | History | Current | Shuffled | Lag-one extrapolation | Zero |
| --- | --- | --- | --- | --- | --- |
| 209011 | 0.126828452 | 0.302759588 | 0.270597045 | 0.284325733 | 0.290091957 |
| 209061 | 0.100734029 | 0.273148980 | 0.205521875 | 0.231889446 | 0.280201487 |

History improves 58.11%/63.12% versus current and 55.39%/56.56% versus lag-one,
beating all planning controls on 16/16 paths. Both one-step and planning gates
pass on both roots. Cached bridge target-rate MSE changes from Stage-29
0.480667313/0.578549610 to 0.014514630/0.021183745; physical means stay identical.
Totals: 6656 fit/3328 fresh query/3328 bridge rows, six fits/multi-RHS solves
(180 RHS), 76864 tape points, zero environment steps or controller/physical
updates. Independent server-side label/forecast/scale/metric recomputation
matched. Retrieved 371330 bytes of JSON; raw arrays remain remote. Retain the
failed 56-row preflight planning gate.

## Next Step

Freeze motion inference; validate causal keep/renew plan response with full
lower feedback and equal upper-call budgets before deploying a learned trigger.

## Limitations

Two-root synthetic forecasts are development evidence, not control reward or
calibrated uncertainty. Other channels and the reused-query bridge are diagnostic.
Shared-latent interference is not isolated. Stage-9 remains the performance reference.
