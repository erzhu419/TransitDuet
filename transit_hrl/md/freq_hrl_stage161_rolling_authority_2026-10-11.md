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
t141969-t141978 DONE on node001/004/005/006. All 200 episodes and 120 source
control reproductions qualify; all 12,276,000 native ticks completed.

| Rolling forecast minus window forecast | Root397 | Root401 |
| --- | --- | --- |
| Cost | +0.179531 | +0.107845 |
| Restricted wait, min | -0.82495 | -0.68220 |
| Fleet cost component | +0.250000 | +0.166667 |
| Nonfleet cost component | -0.070469 | -0.058822 |
| Mean peak fleet | +0.60 | +0.40 |
| Cost wins / 20 scenes | 4 | 9 |

Unlike Stage160, waiting changes are material, especially high-noise/burst
regimes. The full reallocation nevertheless worsens cost and native intrinsic
reward on both roots; headway CV and unserved rate also rise. Reversing the
forecast is much worse (cost +1.431080/+1.269520; wait +8.77805/+7.83795 min).
Wider authority has a real service/resource tradeoff, not a net success.
Next: a preregistered allocation-strength frontier (0, .25, .5, .75, 1), before
training an adaptive strength policy or changing the physical objective.

## Limitations

This tests an executable causal forecasting plan, not a learned upper or complete
Freq-HRL. Development scene/root reuse is not independent confirmation. Wider
authority can worsen fleet peaks or service, and must earn a physical advantage;
moving window endpoints alone is not a success criterion.
