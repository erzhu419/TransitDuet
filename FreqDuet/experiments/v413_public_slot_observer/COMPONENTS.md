# V413: Remaining-Time Component Readiness

Read-only task `t215005` recovered actual components for all 81 error cases.
Each component sum exactly equals actual return time minus query time.
The following means are seconds per error observation, not independent trips:

| Subset | Actual travel | Service and PREPARE | Holding residence | Total |
| --- | ---: | ---: | ---: | ---: |
| 30 misses | 358.90 | 23.33 | 8.77 | 391.00 |
| 51 false alarms | 212.16 | 36.25 | 17.24 | 265.65 |

Service includes latency until command submission; holding residence retains
the mandatory tick for a zero command. Requested future H is separately
225/810 s in the miss/alarm subsets, versus 263/879 s of physical residence.

The detached model-output recorder passed five server checks: query clipping,
zero ticks, equality of existing public-slot predictions, first-return sum
closure and restoration after unsupported input. It reads the frozen model's
event output; neither the model kernel nor the live controller was edited.
The first import-only failure `t215004` is retained.

**Remaining step.** Prediction components were not saved in the original
archives. Actual component sizes alone cannot attribute the return error.
Wire this recorder as an additional sidecar and propose one extra passive
path on old day `2520208`, requiring exact old streams, endpoints and public
slot records plus component closure. Real-path integration/equivalence is
not yet complete; no such path, new day, fitting or policy change was started.

[Small server result](review/component_audit02/results.json)
