# Stage-28 Plan-Hold Result

Frozen implementation `f455dda8d0`; full tasks `t101666/101667` completed
on node004/node006. Each root has 320 fit/160 evaluation pairs, three
ridge fits/15 scalar solves and 658800 new steps. Total: 960 pairs,
1317600 new steps, zero controller reconstruction or updates. Cached
iterations 343/287 reproduce factual return/ISE within 1.03e-12.
Both roots fail both frozen scientific gates.

| Root | Gross MSE history/current | Settled MSE history/current | Settled ISE benefit vs current | vs shuffled |
|---|---|---|---|---|
| 209011 | 0.033868 / 0.032286 | 0.096668 / 0.092566 | +0.011513 | -0.012737 |
| 209061 | 0.036326 / 0.035334 | 0.106516 / 0.104229 | -0.007188 | -0.032407 |

History settled MSE is 4.43%/2.19% worse than current and 9.94%/4.90%
worse than zero. Current wins both gross and settled MSE on 11/16 paths.
Mean gross ISE benefit at 100 steps is +0.095070/+0.092281, with one extra
early call. Equal-call settlement changes it to +0.013162/-0.023526.
These continuation effects do not establish a history-based trigger benefit.

Independent recomputation matches all 320 evaluation keys, arm schedules,
executed call counts, curves, root/path metrics, fits and step accounting.
Retrieved 1236404 bytes of full JSON (plus 48727 preflight bytes); raw
sequences, step costs and controller weights remain on the server.

Next: qualify a causal action-conditioned state/plan-response model, separating
exogenous forecasting from physical control response before deploying a trigger.

## Limitations

Fresh paths reuse two controller roots under a new block-level continuation
budget. This is development evidence, not independent confirmation or a
domain-general performance result. The linear-head failure does not establish
that all temporal representations lack control value.
