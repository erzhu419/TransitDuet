# V115: causal funded holding quota

At each reached decision, issue the current quantized bilateral025 authorization.
Per trip: `balance = issued - completed_actual_holding - expired`, never negative.
Only the current adjacent/same-predecessor innovation enters the request:
`raw = 0.25 * D - 0.75 * delta_R`. Saved credit caps permission, not demand.
Choose an original feasible bin no larger than current positive balance error or
available credit; ties shorter. Balanced gaps or broken references expire credit.
Pending execution blocks further authorization on that trip. Debit actual holding
at departure and check every simulation step's physical holding counter.

First run unit tests and eight consumed V113 smoke paths on server nodes. Require
every actual prefix within unexpired issued authorization, nonpositive innovations
never above the current baseline command, and zero holding at balanced gaps.
Reproduce five cached controls including full timing; test funded sampler invariance
and actual saving/reuse on both archived actors. No fresh day before that receipt.

Then one fixed 72-path independent pilot: actors 12011/933013, days 2151001--2151004;
native B200/preserve three paired sampler repeats each, bilateral025,
bounded_innovation and funded_quota one deterministic path each per actor/day.
Average samplers, use existing 5000-draw crossed actor/day paired intervals.
Primary must pass the original passenger-time improvement and CV2/readiness/
completion no-harm gates against all four controls. No gain refit or ablation promotion.

Authorization is relative to shadows on visited states, not the separate bilateral
trajectory. This is not evidence of frequency separation or external fixed-upper
superiority. Models, upper, dual and main remain frozen; all computation through
scheduler on node001--node006. Only small markers and final statistics return locally.
