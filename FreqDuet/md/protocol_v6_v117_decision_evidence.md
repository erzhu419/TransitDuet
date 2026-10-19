# V117: read-only decision evidence on V115 paths

Read the eight consumed funded trajectories only. Reconstruct
`F = release_gap + executed_holding_gap + R` using the saved predecessor,
and `ready = predecessor_departure + F`. Recover the saved bilateral estimate
as `Ghat = 2D + F`; fallback target is `T = D + F`, not a follower estimate.

For same-predecessor adjacent transitions with unchanged feedback mode,
decompose the change in quarter feedback exactly. Bilateral residual contribution
is `-0.125 delta_R`, fallback is `-0.25 delta_R`; the additional response is
`-0.75 delta_R` when D is positive. Compare net D movement, other coordinates,
and extra response on V116 useful/harmful negative cuts and quota-withheld extras.
Algebraic overlap is not itself a verdict about derivative feedback or causality.

Reproduce V116 cut support and geometry labels. Compare estimated local action
signs with next completed service ready/departure labels. V115 lacks predicted
follower identity: report service-neighbour discrepancies, not identified ETA
accuracy. Exclude unresolved labels with explicit observed counts. Preserve mode
switches, reference resets, boundaries and predecessor mismatches separately.

Tests and analysis run through scheduler on node001--node006; models, actors,
dual and main policy stay frozen. No replay, new days, fitting or threshold search.
Retrieve only the final summary. Decide the next allocation change from this
diagnosis, not from a presumed benefit of restoring every removed action.

## Outcome and decision

Source `6e4111e446`, scheduler `t189888` on node006: six tests and 41,600
decisions pass; V116 labels match exactly. Only 139,869 bytes returned locally.
Extra delta_R response is six times the bilateral residual term already in
quarter feedback (three times in fallback). Net feedback and extra response
agree in 54.7% of stable negative-cut transitions, versus 76.8% for quota extras;
agreement rates are similar in beneficial/harmful restoration cohorts.

Cell-weighted predicted-minus-next-service-ready bias, beneficial/harmful
restoration groups: negative cuts +19.09/+120.29 s; quota extras -2.65/+93.02 s.
All eight cells show the same ordering. The bilateral estimate predicts every
restoration beneficial; ready-to-departure timing flips the sign in only
6.77%/4.82% of eligible negative/quota cases. Post-hoc groups partly select on
this discrepancy, and missing follower IDs prevent identifying pure ETA error.

Do not retune quota or add a D-direction gate: those predictions cannot separate
the two restoration groups. Before changing allocation, capture predicted
follower identity and its action-ready label on an exact, passive replay of
consumed paths, then verify action-difference signs for the same object. That
replay is a proposed next stage, not launched here. No new training or main change.
