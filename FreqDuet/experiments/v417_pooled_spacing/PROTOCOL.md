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

**Result.** `t215056`--`t215060` finished on node004 without retries.
Four new paths completed the factorial with the 12 paid V416 paths. All
four day receipts passed the roster, population, and 5,200-event pairing
checks. Lower is better; entries are equal-day means.

| Metric | Guard under pooled | Guard under hourly | Hourly minus pooled, both guarded | Interaction |
| --- | ---: | ---: | ---: | ---: |
| Journey (s/person) | +0.381 | -5.154 | -4.336 | -5.534 |
| Readiness delay (s/trip) | -0.416 | -4.886 | -8.558 | -4.470 |
| Public launch delay (s/day) | -109.0 | -1280.25 | -2242.25 | -1171.25 |
| Actual arrival CV2 | -0.00386 | -0.00417 | +0.00496 | -0.00032 |
| Actual dispatch CV2 | -0.00426 | -0.00405 | +0.00574 | +0.00021 |
| Mean positive gap excess (s/gap) | -0.943 | -0.635 | +0.641 | +0.309 |
| Holding (s/trip) | -0.057 | +0.076 | -0.568 | +0.134 |

The guard points toward better regularity with either clock, but both
pooled-guard CV2 intervals cross zero. The journey interaction is negative
on all four days, with a descriptive interval of [-11.63, -1.84] s/person.
It mainly comes from waiting, not a uniform change in holding. With both
arms guarded, hourly readiness improves on all four days (interval
[-14.99, -4.16] s/trip), but arrival CV2 worsens on three and the maximum
dispatch gap is longer on all four (+21.75 s on average). The guarded
journey comparison improves on two days and its interval crosses zero
([-11.16, +1.01] s/person). Public delay and readiness measure the same
lateness on different scales in this fixed-roster comparison.

The pooled guard was active on 7,585 of 20,800 queries; 65 queries had at
least one veto. These are not counts of selected actions changed by the
guard. All days remain in the analysis.

**Decision.** Retain the conditional journey signal and the readiness
versus regularity trade-off. Do not treat the two guarded arms as
interchangeable or promote either: V416's failed arrival-CV2 gate stands.
Stop tuning the current ETA and guard family on these four development
days. Any independent study should either predefine an operationally
justified trade-off or test a distinct dispatch-spacing mechanism, not
retroactively change the original nonharm criterion.

[Four-day contrasts and intervals](review/aggregate/results.json)
