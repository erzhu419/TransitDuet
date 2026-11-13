# V413: Running and Holding Error Sources

Read-only task `t215013` reproduces the prior component counts and errors on
the accepted `2520208` archive. No model rerun or new traffic. Full-link
predictions are reconstructed using the frozen strict completed-prefix mean;
the current partial link is the remainder of the recorded running total, not
a separately observed model forecast. The split reconciles all 232 matched
advance slots and 214 at-clock slots.

| Advance error subset | Running error | Future full links | Current partial link |
| --- | ---: | ---: | ---: |
| 3 missed late launches | -407 s | -341 s | -66 s |
| 10 false alarms | +760 s | +759 s | +1 s |

Signed errors are predicted minus actual. Current-link errors can cancel
within a group. On the 232 advance matched cycles, substituting observed
running time in the unchanged launch-clock identity removes 12 of the 13
original classification errors but introduces one. Substituting holding
alone removes none and introduces one. These substitutions use future labels
and do not represent executable forecasts or control effects.

At public slot 225, the 212-minus-214 return gap is -38 s forecast and +19 s
actual. Replacing only running gives -32 s; replacing only holding gives
+7 s; replacing both gives +13 s. For bus 10's later chain, the running split
for slots 236/249/260 is -174/+138/-54 s on complete links and +13/0/-135 s on
the current link. No single link or additive ETA shift explains all three.

**Next.** Prioritize a detached running predictor for complete future links,
using only completed link observations and accounting for when the bus is
expected to enter each link. Keep the partial-link and holding continuation
unchanged in that first comparison. Test on independent days before any
controller change. Treat the slot 225 holding result as a separate mechanism,
not evidence for a global rank or holding gate. This is one consumed day.

[Small server result](review/continuation_audit/results.json)
