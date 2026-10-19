# V127: passive same-vehicle turnover chain

Replay only actors 12011/933013 x consumed days 2231001/2231004 x original
bilateral025 and split_bilateral025: eight paths, no new scenario or control.
Use original V124 bilateral classes, bins, predictor and frozen actors/dual.
Observe actual terminal returns, first visibility in the next dispatch phase,
selected vehicle/prior trip, sticky eligibility, actual launch and denied attempts.
Split launch into eligibility, selected-vehicle availability wait and remaining
post-availability wait; split prior return changes into prior launch, holding and
other elapsed. Keep initial stock and final unobserved visibility distinct.

First run three tests and both paths for actor 12011/day 2231004. Require all
original endpoints/accounts, full timing arrays and stop holding commands exact;
also reproduce the split arm's full decision/evidence trace. Only then replay the
other six paths. Report paired actor/day clock accounting, trip 16 and largest
positive/negative interval contributions; full turnover records stay remote.
These are consumed diagnostic paths, not confirmation or causal mediation.
Selected-vehicle availability is not first possible fleet-wide availability.
CPU conda on node001--node006, one worker per scheduler job; manual server
placement allowed when needed. No training, policy switch, or local rollout.
