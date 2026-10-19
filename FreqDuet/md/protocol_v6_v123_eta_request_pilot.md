# V123: matched streaming-ETA request pilot

Freeze V122 online split ETA, 3600 s strict completed-prefix means and all raw
fallbacks. Compare funded_quota, split_request and bilateral025 with unchanged
B200 actors/upper/dual and original seven-bin masks. No training or main switch.
Split replaces only request-side follower gap and the resulting balance/expiry;
raw bilateral025 on each visited state still issues credit. Innovation gain .75
and feedback gain .25 remain fixed; actual holding cannot borrow future credit.
A nonpositive innovation cannot exceed the corrected request-side shadow. The
raw issuance shadow and corrected request shadow are recorded separately.

First run V115/V122/V123 tests and six consumed V115 paths (two actors, first
old day, three arms). Require exact cached raw/bilateral endpoints and timing,
exact raw decisions, exercised corrected requests on both actors and conserved
every-step physical quota. No fresh labels before this receipt.

Then actors 12011/933013 x days 2231001--2231004 x three deterministic arms:
24 full-day rollouts, paired demand/sampler seeds, all through scheduler on
node001--node006. Report original passenger/journey, actual arrival CV2,
readiness, completion, realized holding and issued/spent/expired authorization.
Use existing 5000-draw crossed actor/day intervals and separate actor/day views.
Against each control, passenger-person-time CI upper must be < 0 and actual
CV2/readiness/unlaunched/incomplete upper <= 0. Report all three contrasts once;
no gain search, day selection, widened margin, or automatic promotion.
Only smoke/shard receipts and final summary return locally; raw traces,
timing arrays and checkpoints remain remote. Two archived actors/four fresh
days support a small structural pilot, not frequency or fixed-upper claims.

## Execution

Source `1046c64e3f`, smoke `t189938` on node006: 23 tests and all six consumed
paths passed. Both original controls reproduce cached endpoints/timing exactly;
raw funded decisions are exact, and both actors exercise corrected requests
without physical-prefix overspending. Only the 254-byte smoke receipt returned.
Formal shards `t190339--t190343`: `[0,4), [4,6), [6,12), [12,17), [17,24)`,
4/2/6/5/7 workers, planned on node001/002/004/005/006. Aggregate `t190345`
waits for all five small local shard receipts. Main and training remain frozen.

## Outcome and next decision

All 24 paths and aggregate exited 0; five rollout shards have zero retries.
Split minus funded: journey -0.187602 min, readiness -15.01 s, holding -18.14 s/trip,
CV2 -0.002327 [-0.004276,-0.0000028]; original joint gate passes narrowly.
Minus bilateral: journey -0.264877 min, readiness -21.45 s, holding -28.85 s/trip,
CV2 -0.000564 [-0.003418,0.004107]; gate fails. Both actor means improve; all
eight cells improve journey/readiness, 7/8 and 6/8 improve CV2 respectively.
Both bilateral failures are day 2231004. Audit `654c2602a6`, `t195158` on node002,
exit 0: two tests, 24 actual-CV2/holding endpoints and quota totals reproduced.
Issued/executed/expired/remaining s: funded 330455/305745/15910/8800;
split 332605/267730/45800/19075. All split expiry is at balanced gaps, not resets.
All 27118 applied forecasts have same-vehicle labels and every cell's MAE improves.
Day 2231004 variance +815.46 s2 versus bilateral; normalization adds only 0.000137
of CV2 +0.006294. Departure arrival-time/current-action terms are +0.006072/+0.000046.
Covariance/local-restoration diagnostics are descriptive, not causal mediation.
Only 45,047 + 95,301 bytes returned; no new rollouts, training or main switch.

Next fill the missing split-ETA + bilateral025 arm on the consumed eight-cell
grid, closing raw/split forecast x bilateral/funded. Distinguish corrected
feedback from innovation/quota/expiry interactions before fresh confirmation;
keep gains, training and main frozen. No joint-dominance claim or promotion.
