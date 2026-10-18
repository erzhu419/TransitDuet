# V108: passive timing diagnosis of the frozen quarter-gain rules

V107 confirms bilateral service gains and CV2 improvement versus B200, but
not CV2 nonharm versus preserve. Dispatch CV2 improves against preserve while
four day means have worse arrival CV2. Diagnose propagation before any new
controller or training design; do not tune gains or reopen the V107 gate.

Replay the first V107 actor from each archived training source (12011, 933013),
all sixteen consumed V107 days, and all five unchanged controllers. Retain three
paired samplers for native/preserve and one path for each deterministic rule:
288 total paths. Selection uses source order, not outcome ranking. Every replay
must exactly reproduce its cached V107 metrics, action counts, coverage and
physical contract. The existing observer already records actual launches,
executed holding since launch, arrivals and departures; no controller changes.

Use actual predecessor order to split each headway into release gap, cumulative
executed-holding difference and other elapsed time. Report covariance allocation,
the six variance/cross-covariance terms, mean normalization, service boundaries
and changed predecessors. Decompose arrivals and departures separately; inspect
all stop/direction groups and existing four-hour bands, retaining support counts
and reporting unmatched bands. Local CVs are not added into the global CV.

Average samplers before global paired crossed actor/day intervals. Global timing
identities must reconstruct actual CV2 changes. Report per-actor and per-day
means and descriptive stop/band changes; no thousands of local significance
tests, outcome-dependent filtering or automatic gate. Raw timing primitives
stay server-side, only a small JSON summary returns. Smoke runs tests plus ten
cached paths (five controllers, two source-order actors) before formal shards.

Decision: if variance changes arise in holding/other-elapsed covariance despite
better release timing, investigate downstream allocation/feedback before new
training. If mean normalization dominates, keep raw variance and service-span
effects explicit. Neither decomposition alone is causal mediation. A new
control proposal requires separate development and independent validation.

## Limitations

This is a two-actor consumed-day mechanism audit, not new confirmation, a fixed
upper baseline comparison or evidence of learned frequency separation. Actor,
critic, dual, main policy and original joint gates remain unchanged. Tests,
replays and aggregation run only through scheduler on node001--node006 using
the isolated CPU environment; the original TransitDuet project stays read-only.
