# V159: clocked joint residual archive

Completed development extraction `t204596` (`6b5dfe58a8`) and independent
readback `t204600` (`944df2d9f1`), node006, eight workers. All 384 actor/day
pairs / 768 archived paths reproduce V158 moments from the written records.
Six extraction tests and four readback tests passed.

Of 3,993,600 queries, 3,343,763 have joint running/service/post-ready residuals;
2,735 retain opaque full residuals and 647,102 lack valid follower evidence.
Identity, query/label clocks, component sums, episode weights, saved observable
state and strict completed-prefix support passed. The 2,787 same-clock ready
states have earlier callback order. Signed errors remain prediction minus truth.

Metadata gaps: 35,462 joint-current-residence queries lack their conditional
support log; shadow service calls are not separately exported (1,695 queries
change service mean); follower-validity is not an exported state field.
All policy snapshots are later than the query, so query-time load is absent,
not backfilled from those snapshots.

Next: recover validity/provenance and conditional support as read-only metadata
sidecars, preserving the residual files. Keep late-snapshot load unavailable at
query time. Independent calibration and confirmation still need distinct unused
day cohorts, disjoint from prior fitting and consumed development/validation,
with complete frozen-deployment archives and outcome-use provenance. None is
certified in the 45-root/78-group inventory; V112's `p0`-`p15` are old diagnostics.

V151/main, V146 (3s/0s), the 3600s window and 0.1/0.1 tails remain unchanged.
No new forecasts, replay, training or interval fit during readback. Raw records
stay server-side; only small receipts return locally.
