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

## Run Record

Frozen source `7d664ad09d`. Smoke `t183574` passed on node004: eight focused
tests and ten cached V107 paths; these paths count toward the 288, not extra
confirmation samples. Formal `t183575--t183579` covers 0--28/28--90/90--173/
173--221/221--288, with 14/31/42/24/34 workers planned from live free CPU.
Aggregate `t183580` waits for those five actual shard markers. Only the
169-byte smoke marker returned before formal dispatch; raw primitives remain
on the servers. All tasks are restricted to node001--node006.

## Outcome

All 288 paths and aggregate t183580 completed with exit 0. All cached V107
endpoints match; no unmatched scopes. Only the 805266-byte final JSON returned.
On these two actors, bilateral minus preserve retains journey -1.1807 min
[-1.3963,-.9984] and readiness -355.50 s [-418.21,-295.26]. Arrival CV2 remains
inconclusive: -.000711 [-.005138,.003814]; raw variance -401 s2 [-1027,235].

The exact arrival CV2 allocation is release -.007283, realized holding
-.003263, other elapsed +.007664, mean normalization +.002170. All four
allocated-term intervals exclude zero. The other-elapsed diagonal alone is
unresolved (+.002160 [-.001959,.006082]); holding/other cross changes by
+.015203 [.008372,.021801]. This is covariance accounting, not evidence that
exogenous travel noise itself increased. Full arrival/departure mean gaps
shorten by 7.046/7.114 s, entirely through earlier last-service boundaries;
first boundaries and full-day sample counts are unchanged.

Arrival bands 0/1 have raw variance +577/+1027 s2, bands 2/3 -2398/-1283 s2;
band sample counts differ as service moves across clock boundaries. At forward
direction stops 1/20, raw variance changes -1368/+1292 s2; reverse direction
stops 20/1 change -3826/-121 s2. Nearly all full-day predecessors are unchanged.
Thus normalization is real but not the whole problem: downstream service and
running-time covariance need separation. The two-actor audit does not replace
V107's ten-actor confirmation or reopen its gate. Next: V109 reads existing
timing primitives to split nonholding stop time from remaining elapsed, with
zero new rollouts; no gain search, training or main-policy change.
