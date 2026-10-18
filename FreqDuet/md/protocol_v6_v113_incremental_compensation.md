# V113: departure-settled incremental compensation

## Frozen design

V112 reproduced all 16 consumed paths. Its repeated negative cumulative residual
responses persisted even when completed follower evidence did not support holding.
V113 replaces the level response, without training or main-policy changes.

For adjacent stops with the same observed predecessor, use the change in cumulative
nonholding residual: `increment = -0.75 * (R_current - R_previous)`.
The first observation, missing evidence, skipped stop or changed predecessor only
establishes a new reference and expires any outstanding budget.

Add increments to a per-trip extra-compensation target. At the actual completed
departure, debit executed holding minus the quantized quarter-gain shadow on the
same state. Requests are never booked as execution. This is actuator accounting,
not a causal attribution of extra holding versus a different trajectory.

Request `0.25 * D + outstanding_budget`, using original feasible bins
`[0, 5, 10, 15, 20, 30, 45]`, nearest-bin ties shorter. Select only bins no larger
than positive current balance error `D`; at `D <= 0`, select zero and expire
outstanding budget. No old shock may reappear after balance without a new increment.
The environment mask and physical action permissions are unchanged.

The mechanism ablation uses the same gains and gap bound but expires outstanding
budget before each new increment, rather than retaining it across adjacent stops.

## Execution order

1. Server unit tests: persistent multistop shocks, balanced gaps, quantization,
   actual rather than requested execution, reference resets and budget conservation.
2. Eight consumed-day smoke paths: exact V110 bilateral controls for both actors;
   native/preserve for one actor; both candidates with two sampler seeds.
   Require exact cached endpoints/timing and deterministic candidate receipts.
3. Only after smoke: actors `12011, 933013`, new days `2131001..2131004`.
   Native B200 and V96 preserve each have three paired sampler repeats;
   bilateral025, ledger_incremental (primary) and bounded_innovation each one.
   Total: 72 full-day evaluations, with actors, upper, dual and physical protocol frozen.

Average samplers within actor/day, then report existing 5000-draw crossed actor/day
paired intervals and per-axis directions. The primary joint diagnostic requires
passenger-time CI upper < 0 and actual CV2, readiness, unlaunched and incomplete-trip
CI uppers <= 0 versus all three controls. Report the ablation without winner search.
Failure closes this fixed candidate; no gain fitting or adaptive day selection.

## Scope

Two archived actors and four independent demand days provide a small structural
pilot, not admission. The balance bound uses the existing follower estimate and
does not guarantee full-day regularity. No external fixed-upper comparison or
frequency-separation claim. All computation and raw traces remain on server nodes
through scheduler; local retrieval is limited to small markers and summaries.
