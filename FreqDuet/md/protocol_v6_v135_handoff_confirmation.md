# V135: frozen handoff, new-demand-day confirmation

Freeze the V134 candidate imported from source366cf193a8; both arms use the
same archived B200 weights, original gains/bins and NO added terminal coordinator.
Primary: ten archived actors excluding12011/933013 x sixteen new simulator
demand seeds2351001--2351016 =160 matched pairs. Secondary: the two development
actors on those days =32 pairs, reported separately. No training or bias refit.

Compare original component ETA with the fixed launch-handoff ETA. Each pair uses
the same scenario and sampler seed, one evaluation per arm:384 rollouts total.
Five focused tests and exact reproduction of four consumed V133/V134 no-terminal
paths precede any new-day evaluation. Six shards x32 pairs run in one campaign
across node001--006 using the existing independent CPU environment.

Primary gate: crossed actor/day bootstrap (existing5000 draws); upper95% CI
below0 for passenger total time, actual arrival CV2 and inclusive published-clock
launch delay, plus no extra unfinished/unlaunched trips. Separately report the
weaker passenger-improvement/CV-delay-nonharm result, all actor/day axes, zero
intervention cells and negative cells. Neither secondary cohort nor a pooled
average substitutes for the primary held-actor gate. No missing arm is zero-filled.

All weights and the main deployment stay unchanged regardless of the result.
Keep pair receipts, compact timing arrays and launch traces on the server; fetch
only smoke/shard markers and the small final summary. These are simulated demand
days and candidate-held-out archived actors, not field dates or new training runs.
