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

## Outcome and next decision

Source `4c17d07c64`, scheduler `t188762`, node006: seven tests and all 16 trace
reconstructions pass, covering 83,200 executed actions and 41,600 matched stops.
Only the 104,729-byte summary returned locally; no new rollout was run.

Ledger minus single-step holding is 23,490 s: the same-ledger-prefix carry response
is +22,845 s, with +645 s between-path prefix difference. The carry response adds
18,145 s net on decisions where the current innovation requests a reduction;
only +1,275 s net occurs with zero current innovation. The problem is not confined
to constant disturbances, and this arithmetic is not a passenger-loss attribution.

For single-step response, signed raw increments sum to +366.75 s, but after
conditioning execution on positive balance error, effective increments sum to
+151,589.25 s. After actuator transformations, same-state excess is +95,940 s.
Thus near-zero raw incremental demand does not imply zero-mean executed holding.

Do not restore compensation debt or retune its gain. Next design a separately
frozen, causal funded-quota response: accrue the current quantized quarter-rule
authorization, spend only completed actual holding, never borrow future credit,
and use only the current innovation for the request. Balanced gaps expire credit.
First verify that every prefix's actual holding stays within issued authorization
and that a negative new increment cannot be overridden by old compensation debt.
This bound is relative to shadows on visited states, not a guarantee versus a
different bilateral trajectory; an independent joint-service pilot is still needed.
