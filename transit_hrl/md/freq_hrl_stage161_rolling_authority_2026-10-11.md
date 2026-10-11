# Stage161: Rolling-Horizon Planning Authority

Stage160's learned policy is nearly indistinguishable from its training-only
constant control. The existing six-trip representation fixes every window's
endpoints: it cannot transfer service time between windows. Test wider causal
authority before changing another optimizer or expanding seeds.

At each first uncommitted trip, forecast the remaining direction horizon using
the CURRENT harmonic tracker, project preferences onto 240-480 s intervals
with the exact remaining time budget, and commit only the next six trips.
Future tentative times set native callback eligibility; later forecasts may
revise only uncommitted departures. Daily first/last times, trip count, native
lower weights, fleet target, evaluation horizon and 44 upper decisions stay fixed.
The new bounded-simplex projection avoids a single saturated interval freezing
all later reallocation; the old allocator and previous protocols are unchanged.

Conditions: original window forecast/nominal, rolling nominal, rolling forecast,
rolling reversed. All window controls and rolling nominal must reproduce Stage160
exactly, including actual physical outcomes. All four scenes per regime/root;
no selection of the OOD scenes that drove earlier fleet jumps. Report cost,
waiting, headway, unserved and fleet/nonfleet cost contrasts separately.

Two frozen lower policies, five regimes: ten scheduler cells, 20 episodes each,
200 episodes/12,276,000 native ticks, zero training updates. Node001-006 unpinned,
1 CPU/3 GB per task. Only compact result JSONs sync locally; raw CSVs and existing
checkpoints remain remote. No new native training in this stage.
110 focused tests pass, including bounded projection without horizon freezing,
real native-dispatch timing/goals, irrevocable commitments, tail windows, source
reproduction, worker control qualification and analyzer/scheduler budget checks.

Run `native_transit_rolling_authority_stage161_frozen_20261011_r1`:
t141969-t141978 launched on node001/004/005/006. First-scene qualification
requires window forecast, window nominal and rolling nominal to reproduce
Stage160 before extended authority runs. Final physical performance is pending.

## Limitations

This tests an executable causal forecasting plan, not a learned upper or complete
Freq-HRL. Development scene/root reuse is not independent confirmation. Wider
authority can worsen fleet peaks or service, and must earn a physical advantage;
moving window endpoints alone is not a success criterion.
