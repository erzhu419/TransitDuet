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

Preflight completed on node004 with exact source baseline reproduction,
unchanged networks and 61,380 ticks. Root217/low-noise critic LCB preferred
+15 seconds at 83.2% of recorded states and +30 at 16.8%, while learned mean
was +0.38 seconds. This is a value/actor discrepancy to investigate, not proof
that either fixed goal improves physical service. Legacy global transitions
had median duration 180 seconds and a final 14,400-second clearance interval;
physical-only has no interval wait credit, so its zero tail share is not a finding.

Full scan `native_transit_frontier_stage149_full_20261009_r1` completed as
`t138705-t138708`. All 20 source baselines reproduced exactly; the 100 new
episodes used 6,138,000 ticks and zero optimizer updates, reusing 60 goal rows.
Only code was staged and small JSON was synchronized.

The retrospective per-scene grid gain against zero averages 0.014743/0.023143
for physical-only roots217/229 and 0.013656/0.022287 with service credit.
Three of four deployments have zero as their best fixed grid goal; only
physical-only/root229 prefers +15 (cost gain 0.008962). Physical-only/root217
critic LCB prefers +15 in every regime, but that fixed goal raises average
cost by 0.153442: an extra peak vehicle in low-noise and burst scenes erases
the headway/wait gains. Static critic ranking is therefore not a justified
deployment rule. Near-zero learned goals are not, by themselves, actor failure.

With interval credit, the persistent-shift final 14,400-second clearance
transition carries 52.9-54.0% of wait credit, versus typical 180-second intervals.
These are frozen stress episodes, not proof of Bellman-target clipping during
training. Do not change the RE-SAC critic on that untested explanation.

Next: qualify an actual signed departure-action channel rather than add more
HIRO goal seeds. Inspection found that ordinary native channels query upper
only at nominal launch, so negative offsets cannot advance departures. Stage150
will repair this execution timing in the isolated native copy, reproduce the
unchanged HIRO baseline, and verify advance/zero/delay with frozen lower weights.

## Limitations

This is post-result diagnosis on reused development scenes. The retrospective
grid minimum is an opportunity diagnostic, not a deployable policy or a new
performance result. Same-state critic rankings do not equal full-episode policy
values. A fixed-goal grid does not exhaust state-dependent policies. Any apparent
benefit needs learned, causal execution and independent validation afterward.
