# V114: read-only V113 incremental-budget audit

Read the two deterministic variants on all eight consumed V113 actor/day cells:
16 existing compressed traces and timing records. No replay, fresh day or update.

Reconstruct the frozen per-trip ledger and executed receipts; require identical
commands, timing, accounting and action counts. Decompose same-state command
excess exactly into feedback-shadow difference, effective new innovation,
carried budget, zero/45s clipping, current-balance projection and quantization.
Partition carry and innovation signs; report balanced and persistent cohorts.

Match trip/stop identities across both paths. Split their total holding difference
into the carried response on the ledger's observed prefix and the remaining
between-path prefix difference. These are action identities, not a causal split
of passenger-time or CV differences. Per-decision passenger loads were not logged.

All tests and analysis run through scheduler on node001--node006. Only final
summary/status.json returns locally. V113 primary stays closed, the ablation is
not promoted, and actors/critics/dual/main remain frozen.
