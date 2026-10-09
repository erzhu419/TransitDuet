# Stage149: Frozen Upper Opportunity and Value Diagnosis

Stage148 recovered physical goal execution, but learned upper remained near
zero. Large +/-60 goals have competing wait/headway and peak-fleet effects.
Before changing RE-SAC, distinguish poor value learning from a near-optimal
zero goal in this fixed-dispatch hierarchy.

Freeze physical-only and physical-plus-credit deployments at episode299,
roots217/229. Query the native critic at identical upper states for goals
[-60,-30,-15,0,15,30,60] seconds, preserving physical units and its epistemic
LCB. Record interval duration and the final transition's wait-credit share.
First reproduce one original baseline exactly, then reproduce all five regimes
and simulate only missing +/-15,+/-30 goals. Reuse the existing neutral and
endpoint outcome rows rather than rerunning them. No optimizer updates,
checkpoint selection, new seeds or checkpoint downloads.
Twenty-two focused frontier/authority/diagnostic tests passed, covering passive
critic queries, temporal-credit accounting, cached-goal pairing, baseline
reproduction, incomplete-scene rejection and code-only scheduler staging.
Preflight `native_transit_frontier_stage149_preflight_20261009_r1` is registered
as `t138703`, eligible for node001-006 without binding, one CPU and 2 GB.

Next: source-reproducing preflight; then four frozen jobs, 100 new episodes
(6,138,000 ticks), reusing 60 existing goal rows. Only code is staged and small
JSON is synchronized. Inspect the cost frontier, wait/fleet tradeoffs and
critic preferences before deciding on reward/critic repair or action-space work.

## Limitations

This is post-result diagnosis on reused development scenes. The retrospective
grid minimum is an opportunity diagnostic, not a deployable policy or a new
performance result. Same-state critic rankings do not equal full-episode policy
values. A fixed-goal grid does not exhaust state-dependent policies. Any apparent
benefit needs learned, causal execution and independent validation afterward.
