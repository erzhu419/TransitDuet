# Stage-27 History Information Result

Tasks `t101659/101660` completed on node004/node006 at `c1e5e72a44`.
Each root used 320 fit/160 evaluation rows and six fixed unit-ridge probes.
Fourteen tests passed; independent label and metric recomputation matched
within float32 rounding. These are fixed-lag velocity probes, not Stage-26 GRUs.

Future-target displacement-rate MSE, averaged over 10/25/50 steps:

| Root | History | Current | Shuffled | Lag-one extrapolation | Zero |
|---|---:|---:|---:|---:|---:|
| 209011 | 0.106076 | 0.303649 | 0.285230 | 0.145536 | 0.346022 |
| 209061 | 0.127927 | 0.308515 | 0.259161 | 0.177694 | 0.379053 |

History improves 65.07%/58.53% versus current, on eight/seven evaluation
paths. It beats lag-one extrapolation on the averaged endpoint, but that
simple control remains slightly better at the 10-step horizon on both roots.

History decision mean local ISE benefit over each timing control:

| Root | Current | Shuffled | Always wait | Always now |
|---|---:|---:|---:|---:|
| 209011 | +0.000364013 | -0.000203611 | +0.003309136 | -0.001748567 |
| 209061 | +0.000633237 | +0.001496643 | +0.004903929 | +0.000568011 |

Timing-response MSE is 2.79% worse/0.18% better than current. Useful history
forecasting does not yield a stable timing advantage against all controls.
Next: replace five-step timing supervision with genuine multi-duration
keep-old-plan/renew credit, preserving lower feedback and freezing the
planning-resource comparison before sampling new paths. Reuse remote weights.

Retrieved 822,434 bytes of full JSON. Twelve fits/54 scalar solves generated
57,648 exogenous tape points, with zero new environment steps or controller
updates. Preflight scale amplification and its negative result are retained
in the protocol; full training did not exhibit it.

## Limitations

Retrospective reused-root linear probes do not qualify a deployed policy or
isolate supervision as the only cause. Stage-26 remains failed and Stage-9
remains the performance reference.
