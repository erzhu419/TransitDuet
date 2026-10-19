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
