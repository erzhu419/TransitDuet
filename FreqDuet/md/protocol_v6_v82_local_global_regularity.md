# V82: local-to-global regularity diagnosis

V81's fixed candidate failed both unchanged joint gates. It retains stochastic
passenger and readiness benefits versus B200 and improves CV-squared versus raw
V78, but gives back about 42% of passenger benefit. Existing-pair audit reproduces
every primary mean/CI; passenger improves on 29/32 days (largest positive-gain day
8.4%). Both waiting and onboard exposure improve. Three-repeat CV signs remain
mixed on 19/32 days, yet projected-minus-B200 between-day variance is positive
with CI [5.90e-7,6.58e-6]: sampling noise alone is not the only variation present.
No V81 threshold tuning, candidate promotion or new long training follows.

Replay four evenly spaced, already consumed V81 days (1810001,1810009,1810017,
1810025), both original actors, all three stochastic repeats and deterministic
deployment. Fixed B200/raw/projected laws give 32 cells, 96 full-day replays.
This is a descriptive mechanism subset, not a new confirmation or gate test.

Record causal pre-action contexts and the same distributions actually sampled.
Passively log the locked runtime's original prepare-for-action and departure
calls, capturing follower trip identity at the original AVL estimate's instant.
Restore both observation hooks after each episode. After completion, match the
follower-ready and departure events. Separate missing bilateral evidence, active projections and
valid-but-inactive states. Compare each visited state's predicted expected gain
with the same local formula replayed using completed follower gaps; report
coverage, follower error and subsequent follower holding. Completed-gap replay
is not a causal alternative-action value: changing current holding could also
change that follower's future. No completed label enters action selection.

Decompose measured arrival CV-squared into within/between stop-direction and
four-hour groups, retaining additive group contributions. This distinguishes
the state-local departure surrogate from the actual all-day arrival metric;
group contributions are descriptive, not causal credit assignments.

Server smoke tests the decomposition, missing-label handling and exact policy
laws, then replays the first three-policy cell. Every replay must reproduce its
original V81 outcome metrics and action histogram exactly, otherwise stop the
diagnosis. Reuse smoke cell 0 in the remaining scheduler batch, node001-node006
only, then aggregate on-server. Pull only small summaries. Actors, critics,
production dual and main policy remain frozen. Next control design depends on
whether failure aligns with missing coverage, follower forecast mismatch, or
downstream/global propagation; no new controller is fitted in V82.

Initial smoke `t166208` passed four unit tests but stopped before completing any
rollout: the locked optimized runtime lacks the newer Bus/recorder trace fields.
Its four gated shards and aggregate were cancelled. Record the original runtime
calls instead; do not upgrade or modify its physics. Re-smoke the exact three
original endpoints before releasing replacement tasks.
