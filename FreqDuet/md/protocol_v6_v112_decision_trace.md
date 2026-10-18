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
