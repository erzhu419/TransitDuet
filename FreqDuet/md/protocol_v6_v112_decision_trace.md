# V112: passive decision traces on consumed V110 paths

Replay only actors 12037/933083, days 2101001--2101004, bilateral025 and the
closed V110 feedforward candidate, rep=0: 16 already-consumed paths. No new
day, coefficient fit, model update or main-policy change. Two first-day paths
form smoke; formal replay reuses those receipts and executes the remaining 14.

Use the original V110 law and sampler seed. Before forming post-completion
labels or summaries, require exact full result/contract/coverage/feedforward
equality and identical arrival, departure and launch arrays to the cached V110
path. Failure blocks that path and downstream aggregation.

Trace actual ready/policy times, identities, feasible bins, forward/follower
evidence, release/executed-holding/nonholding coordinates, prior executed own
holding and same-state shadow baseline/excess ledger, raw requests and commands.
The frozen runtime exposes follower bus_id, not trip_id; resolve it from the
same prepare-action prefix, never from the later vehicle/trip assignment.
Only adjacent stops with the same predecessor support carried-level versus
innovation comparisons. Track persistent negative levels after prior holding,
especially when the feedback term is nonpositive. Innovation commands are
unexecuted shadow arithmetic, not alternative-policy outcome labels.

After reproduction, resolve the predicted physical follower's actual same-stop
ready time and departure separately; the original ETA targets action-ready,
excluding its future discretionary holding. Report label coverage, timing and
identity mismatch, feedback sign/command differences. Partition command delta
exactly into raw feedforward, clipping delta and feasible-quantization delta.
These overlapping diagnostics do not decompose the causal passenger-time loss.

All tests and replays go through scheduler on node001--node006. Full compressed
decision traces and per-path diagnostics remain remote; only smoke/shard
markers and final summary/status.json synchronize locally. Main stays frozen.

## Run Record

Frozen rollout source `e8792aa58b`; scheduler smoke `t183923` passed seven tests
and both full-path exact replays on node006. Only its 113-byte marker returned.
Formal `t183924` (0--8, node001) and `t183925` (8--16, node002), each eight
workers, launched after that marker; aggregate `t183927` waits for both shards.
Old `t183657--t183660` were cancelled after identifying the frozen runtime's
missing follower-trip field; no old formal shard ran. The fix records the
existing bus-to-trip identity at prepare time and does not alter control.

The separate read-only cross-review reads completed compressed traces only
after the all-path aggregate receipt. It separates repeated positive holding
with nonpositive feedback into missing-forecast and matched completed-follower
cohorts, and checks whether 45s commands persist under completed follower gaps.
No additional rollout or alternative action is executed by this review.

## Outcome And Decision

All 16 paths reproduce every cached endpoint/audit and timing array. Nine tests
pass. Aggregate `t183927` and cross-review `t188745` completed on node006;
review source `09a0f6c5a7`. Only 216,405 bytes of summaries and 429 bytes of
markers returned; all traces stay remote.

FF has 6,757 repeated negative-R decisions with prior holding, nonpositive
feedback and positive commands; 2,888 have no new negative innovation, and 316
use no follower forecast. With matched completed follower labels, 5,784/6,272
still have nonpositive feedback (200,955s holding), occurring in all eight
actor/day cells. Follower-ready MAE is 64.75s bilateral025 / 70.86s FF, but
8,642/9,626 matched FF 45s commands persist using completed gaps (89.78%).
Same-state command excess 6.214s = raw feedforward 1.292 + clipping 4.910 +
quantization .012. This diagnoses net command bias, not causal shares of service
loss; near-zero net quantization does not establish negligible variance effects.

Close V110. Next prioritize executed-compensation accounting and incremental
disturbance response, testing multistop persistence/balanced gaps before any
separately frozen controller pilot. No gain rescue, new training or main switch.
