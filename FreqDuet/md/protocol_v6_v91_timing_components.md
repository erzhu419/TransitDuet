# V91: exact timing and variance components

V90 preserves V89's passenger/readiness improvements but leaves CV non-harm
unresolved. Its positive CV mean-normalization term and largely advanced trip
timing require distinguishing release changes from on-route dispersion. A
completed-follower local formula remains positive on average, so another
forecast correction alone has no demonstrated control justification.

First summarize existing V90 absolute mean headway/raw variance and day-level
local-formula errors on-server. Then passively replay the same 96 consumed
cells under late/no-gain-only laws, 192 full-day diagnostic paths. Capture
actual terminal release, cumulative realized holding since that trip's launch,
and remaining elapsed time at each arrival/departure. The remaining term
includes travel, passenger service and clock effects, not travel alone.

For each path's actual predecessor order, headway equals the sum of release,
holding and other elapsed-time differences. Allocate raw variance by covariance
with these components; their changes plus mean normalization must reconstruct
the original CV change. This is exact descriptive accounting, not causal
mediation. Release includes fleet turnover and reactive frozen upper decisions,
not changes in upper weights. Report predecessor changes and fixed four-hour bands. Separately
reconstruct the mean-headway change from first/last service-span endpoints and
sample count. Do not substitute these diagnostics for the original CV gate.

Smoke tests these identities and must reproduce both actors' V89 endpoints,
contexts, full sampling/holding paths, arrival CV and actual readiness. Stop
on a mismatch. Formal cells must reproduce V89 metrics/action counts/contexts.
Keep compressed primitive timing records on-server to avoid future replays;
only small summaries are retrieved. All tests, replays and statistics use
scheduler on node001-node006. Actor/critic/dual and main policy remain frozen.
No new candidate, gate tuning or controller admission occurs in V91.

The initial smoke compared raw readiness counters with a rounded episode row
and stopped on a 3.05e-8 s difference; its waiting tasks were cancelled. Compare
raw counters directly with the raw simulator readiness metric instead, while
retaining exact original V89 episode-row comparisons. No statistical or
scientific tolerance/gate is changed.

Verified source `8f54a2a06c`; smoke `t169699` passes six tests, both actors'
native/conditional full event and actual-holding paths, original V89 metrics
and contexts, raw readiness, common arrival/departure support and exact
variance/service-boundary identities. Review is read-only consumed-day data.
Shards `t169701-t169704` cover `[0,24)`, `[24,48)`, `[48,72)`, `[72,96)`;
the live planner allocates 24 workers each on node002/004/005/006. Reuse smoke
cells 0/48. Aggregate `t169706` waits for those exact four small markers.
Compressed timing records stay on-server. Formal mechanism results are pending;
main policy, actors, critics, dual and original gate remain unchanged.
