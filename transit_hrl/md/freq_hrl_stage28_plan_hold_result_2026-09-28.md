# Stage-28 Plan-Hold Result

Tasks `t101666/101667` completed on node004/node006 at frozen revision
`f455dda8d0`. Both roots fail both scientific gates.

| Root | Gross MSE history/current | Settled MSE history/current | Settled ISE benefit vs current | vs shuffled |
|---|---|---|---|---|
| 209011 | 0.033868 / 0.032286 | 0.096668 / 0.092566 | +0.011513 | -0.012737 |
| 209061 | 0.036326 / 0.035334 | 0.106516 / 0.104229 | -0.007188 | -0.032407 |

History settled MSE is 4.43%/2.19% worse than current, which wins on 11/16
paths. Mean 100-step gross ISE benefit is +0.095070/+0.092281 with an extra
early call; equal-call settlement changes it to +0.013162/-0.023526.

Independent recomputation matches all 320 evaluation rows' schedules,
call counts, curves and root/path metrics. All 960 pairs, six fits/30 solves
and 1317600 new steps are accounted for; zero controller updates or rebuild.
Cached iterations 343/287 match factual return/ISE within 1.03e-12.
Pulled 1236404 bytes of full JSON; arrays and weights stay on the server.

Next: qualify a causal action-conditioned state/plan-response model, separating
exogenous forecasting from physical control response before deploying a trigger.

## Limitations

Two reused controller roots under a new continuation budget provide
development evidence. Failure of these linear heads does not establish
that all temporal representations lack control value.
