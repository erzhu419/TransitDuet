# V400: independent demand-day confirmation

All 80 confirmation paths and aggregation completed on node004: 16 new demand
days, five arms, eight concurrent shards, zero failed paths or retries. Including
the two old-day engineering reproductions, the physical-start count is 82.
The longest shard took 202.4 minutes. No training or main-policy change occurred.
The aggregate JSON is 58,771 bytes; raw traces and weights remain on the server.

Differences below are candidate minus comparator; brackets are the frozen
day-bootstrap 95% intervals. Journey is seconds/passenger; readiness is seconds/trip.

| Comparison | Journey | Readiness | Actual CV squared |
| --- | --- | --- | --- |
| K4 rho=.25 minus K4 rho=.50 | -1.645 [-3.074, -0.070] | -3.804 [-4.781, -2.915] | +0.005646 [+0.003261, +0.009176] |
| Manual rho=.25 minus manual rho=.50 | -2.396 [-3.771, -0.976] | -4.540 [-5.579, -3.595] | +0.003025 [+0.001410, +0.004741] |
| K4 minus manual, rho=.50 | +0.022 [-0.608, +0.584] | -0.690 [-1.400, -0.048] | -0.000867 [-0.002202, +0.000353] |
| K4 minus manual, rho=.25 | +0.773 [+0.075, +1.458] | +0.045 [-0.620, +0.570] | +0.001754 [-0.000713, +0.005083] |
| K4 minus matched K0, rho=.50 | -0.562 [-1.567, +0.352] | -0.104 [-0.597, +0.436] | -0.000937 [-0.001668, -0.000175] |

The preregistered budget-service claim passes: tightening improves journey and
readiness, with worse regularity. The preregistered harmonic-journey claim fails.
Harmonic regularity improves as a secondary result; it does not replace that gate.
Learned-upper journey superiority is not established at rho=.50 and reverses at
rho=.25. Keep the paper's budget/trade-off claim; narrow the harmonic journey claim.

## Scope

One frozen learner seed and 16 independent simulation demand days; no independent
future replications or retraining-seed confirmation. Cross-zero intervals do not
establish equivalence. All 80 physical/quota checks and 5,200-event paired-path
checks passed. See [the complete aggregate](review01/aggregate/results.json).
