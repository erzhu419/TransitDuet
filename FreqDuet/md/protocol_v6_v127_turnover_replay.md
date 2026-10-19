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

## Outcome and next decision

Source a4aaaef2ca; scheduler t195174--t195178 on node006/004/005/002;
clock audit t195189 on node006. All actual child exits 0, three tests pass,
all eight paths reproduce endpoints, timing and commands exactly. Same-vehicle
prior-trip matches are complete. Only 94,741-byte summary and tiny receipts return.
All paths have zero post-eligibility-and-visibility delay; every observed
return-to-next-dispatch phase is 1 s. There is no extra dispatch-loop delay to fix.
On failing day 2231004, trips 230/232 advance 188 and 48/52 s: their interval
grows 541 to 681/677 s. Earlier returns need not yield more regular releases.
On improving day 2231001, advancing trip 185 restores adjacent intervals from
606/113 to 360/358 s for actor 12011. Eligibility clipping and unequal adjacent
return shifts explain why mean readiness alone is insufficient.
Keep main frozen. Next test executable terminal-release/enroute-holding
coordination in a small matched counterfactual, using live eligibility, actual
past releases and current fleet availability. Establish joint passenger/CV2/
readiness headroom before training or promotion; no additional ETA/quota tuning.
