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

## Run record

Source `536feb4f0e`, scheduler smoke `t188755` on node001: all 13 tests and
eight full-day replays passed. Four cached controls reproduce every result and
all arrival/departure/launch arrays exactly; both incremental variants reproduce
outcomes and settled execution accounting across two sampler seeds. Only the
237-byte completion marker returned locally. Static review corrected an equality
check that included intentionally different sampler IDs; old smoke `t188754`
was stopped before independent labels, with no change to the controller or gates.

After that receipt, submitted one independent batch `t188756--t188760`:
shards `[0,3), [3,23), [23,34), [34,45), [45,72)` with `3/20/11/11/27`
workers, planned on node001/002/004/005/006 from live capacity. Aggregate
`t188761` waits for all five local completion markers and runs on a server.
Only shard markers and final `summary/status.json` may synchronize locally.
All five shards and aggregate completed with exit code zero. Retrieved only the
96,527-byte summary; all 72 full-day evaluations have zero unlaunched/incomplete
trips. Both variants settle 41,600 actions, with zero positive holding at balanced
gaps and zero accounting conservation error.

## Outcome and decision

Ledger versus B200: journey -0.9923 min, actual CV2 -0.00593, readiness -370.35 s;
versus preserve: -0.6805 min, -0.00562, -262.16 s. Both joint diagnostics pass.
Versus bilateral025: journey +0.4350 [0.2796, 0.5389] min, CV2 -0.00853
[-0.01016, -0.00646], readiness +56.36 [48.36, 68.33] s: the third gate fails.

Retaining budget versus single-step innovation adds journey +0.1598
[0.0895, 0.2128] min, holding +11.21 [8.79, 13.82] s/trip and readiness
+20.67 [13.18, 29.54] s. CV2 +0.00056 [-0.00041, 0.00177] gives no regularity
benefit; all eight actor/day journey differences are positive. Close the carried
budget candidate, rather than tune its gain or promote the ablation post hoc.

Single-step innovation also jointly improves B200/preserve, but versus bilateral
still adds journey +0.2752 [0.1927, 0.3357] min, holding +36.64 s/trip and readiness
+35.69 s while improving CV2 -0.00909. No all-control joint improvement is shown.
The subsequent read-only V114 audit covers all 16 existing traces; its record is
in `protocol_v6_v114_incremental_budget_audit.md`. No new rollout, training or
main-policy change is justified by this pilot alone.
