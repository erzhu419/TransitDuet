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

**Passive integration.** One additional old-day path on `2520208` is released
only after the server's zero-traffic entry checks. Reuse the existing V413
parity archive: require all 12 old streams, endpoints and both public-slot
records to match exactly, excluding only query wall time. New components must
match old return clocks and close for each vehicle/scenario. Failure consumes
the one allowance, with no automatic retry or expansion. No new days, fitting
or policy changes. [Frozen allowance](component_parity_protocol.json).

[Small server result](review/component_audit02/results.json)

**Entry and launch.** `t215007` passed the zero-traffic entry checks, including
old public-slot equality during clock progression and nine closed component
records. The single real-path allowance is charged to `t215008`, manually
dispatched high on `node004` and confirmed progressing through 400 queries.
Full-path equivalence subsequently passed: the 12 old streams, endpoints and
public slots are exact, and all 5,818 component records close. No second path
or new-day collection is released. [Entry check](review/component_checks/results.json),
[launch receipt](submission_component_parity.json).

[Accepted observer and old-day component findings](COMPONENT_READOUT.md)
