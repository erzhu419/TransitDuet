# V417: Complete the forecast-by-spacing factorial

Add pooled V406 with V416's unchanged spacing guard on the four consumed
development days 2560001--2560004. Reuse the three paid V416 arms on the
same days. The new arm uses the original pooled event graph, the original
capped controller, and exactly the frozen guard used by the hourly arm.
Keep the slow input, demand, physical fleet, published roster and action
menu unchanged. No additional engineering traffic, tuning or training.

One zero-traffic server check verifies source binding and the paid receipts.
Then release at most four starts, one per day, all scheduler high and
manually dispatched. Failure consumes any started slot, with no retry.
Require the same generated passenger count, public clocks, and 5,200
random-event keys and draws as the paid comparators. Sync only summaries.

Use equal-day contrasts for adding the guard under pooled and hourly
predictions, and for adding the hourly predictor with and without the
guard. The interaction is the hourly guard effect minus the pooled guard
effect. Report all four day effects and descriptive 95% intervals from
the same 256 day bootstrap draws as V416. Interpret journey, actual arrival
and dispatch CV2, positive gap excess, holding and delay together.
The four consumed days are for attribution, not independent confirmation.
V416's failed screen remains in force and no arm replaces the main policy.

**Dispatch.** Preflight `t215055` passed on node004 with zero traffic.
Tasks `t215056`--`t215059` each wrote one startup receipt on node004.
Aggregate `t215060` waits for the four day summaries. All five tasks were
submitted high and manually dispatched, with automatic requeue disabled.
The raw startup receipts retain the V415 runtime identity. No effect readout
is available at release.

[Preflight result](review/preflight/results.json) |
[Factorial submission](submission_factorial.json)
