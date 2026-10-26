# V275: closed-flow evidence and executable continuation

One complete controller revision following the V268--V274 review. Keep V107,
V272, V274 and the original TransitDuet archives unchanged. No actor, dual,
training, fleet, action, timetable, horizon, cost-weight or beam changes.

Demand uses the last three positive-duration, nonoverlapping clearance blocks
with strict completion and availability clocks. Counts include intervening full
visits. Their count/time ratio is a constant-rate future point extrapolation,
not an identified future bound. Unclosed intervals remain censored. Initialize
queue predictions from the last closed anchor and subsequent visible APC.

For each proposal, simulate both its full plan and only its current action with
baseline continuation. Both must satisfy the existing predicted CV2, physical
return, next-launch and public-delay constraints; fallback passenger cost cannot
increase. Minimize the worse passenger cost. Both evaluations share the original
128-simulation budget; beam4, variables6. Record future proposals and their
subsequent actual command receipts, changed requests and cancellation reasons.

Freeze days2075001--2075004 after server metadata non-consumption checks. Three
arms: fixed360+bilateral025, flow/APC+fallback, flow/homogeneous-cost+fallback.
One actor anchor12011, sampler27011001; 12 unique full days, not independent
future replications. Server synthetic checks precede starts. Eight single-process
units maximum, high/manual scheduler, node001--006. Each slot starts once;
failure consumes its slot without retry, replacement or retuning. Raw records
remain server-side; local transfers are small JSON/log receipts only.

Use four paired days and10000 shared bootstrap resamples(seed2750201). Against
BOTH controls require passenger-cost upper95<0 and actual CV2/readiness/public
total-delay upper95<=0, without extra unfinished trips on any day. Report day
values, action counts, executed H, support/fallback, proposal realization and
latency. A failed gate closes this frozen candidate; a pass only permits proposing
independent confirmation, not switching main or claiming frequency superiority.
