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
Report modal action/gain changes separately from distribution means, since
deterministic argmax deployment does not inherit the expectation constraint.

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

Initial instrumentation attempts lacked locked-runtime fields (`t166208`) and
missed departure capture after recorder reset (`t166225`). Their dependents were
cancelled; their original output root contains incomplete diagnostic data only.
The corrected class-method observation has a reset-survival test and requires
actual completed follower labels as well as exact original outcome replication.

Verified source `9a1519183a`, smoke `t166239`: six tests passed and all three
original V81 outcomes/action histograms reproduced exactly. Completed follower
ready/departure labels cover all 4359/4358/4360 bilateral B200/raw/projected
contexts in the smoke. The physical runtime and policy laws are unchanged.
Current output is `protocol_v6_v82_local_global_regularity_completed_events`;
no incomplete old pair is resumed. Shards `t166240-t166243` cover `[0,8)`,
`[8,16)`, `[16,24)`, `[24,32)` on node002/004/005/006 with eight workers each.
Reuse smoke cell 0; aggregate `t166244` awaits the four small completion markers.
V82 mechanism results are pending; no new controller is admitted.

All 96 replays and aggregation completed, exit 0/retry 0; all original V81
endpoints/action counts remain exact. Read-only report `t166273` passed three
variance-decomposition tests and reproduced the source CV means. Only summary
JSON was retrieved (70,804 + 28,411 bytes); raw traces/checkpoints stay on-server.
Stochastic projection: 16.02% missing bilateral evidence; the remaining 83.98%
has complete ready/departure labels. Active-group gap bias is +64.02 s versus
-17.19 s inactive. Active local expected gain is numerically zero, completed-
ready replay gain +8.48e-6, completed-departure gain -1.50e-6; 20.85% of active
states have predicted non-harm but completed-ready replay harm. These are
visited-state formula audits, not causal alternative-action labels.

On these four consumed days, stochastic projected-minus-B200 CV-squared is
+0.000749: +0.000626 from variance at the reference mean and +0.000124 from
mean normalization. Within-stop variation contributes +0.000751, between-stop
means -0.00000150; four-hour within-band variation +0.000721. This locates the
discrepancy but does not identify downstream propagation causally. Deterministic
CV improves on this subset but is heterogeneous by actor and does not revise
the failed 32-day V81 gate. V83 will intervene once and freeze subsequent lower
commands in one control arm, rather than fit another forecast correction.
