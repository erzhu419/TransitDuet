# V214: inbound supply-chain joint counterfactual preregistration

Canonical executable registration: `preregister_freqduet_v214_inbound_chain.py`,
source a572e63a8b; server `protocol.json` sealed before any pilot replay.

## Selection and authority

V211's four consumed B200/bilateral025 contexts. First fresh upper decision in
[7200,8100): currently due terminal queue, no compatible ready vehicle. Commit
the nearest two opposite-direction active trips by current remaining distance,
then bus/trip ID; at least one has a controlled stop at/ahead. No anchor stops
the pilot. Within 900s, reduce each committed trip's FIRST nonzero bilateral,
valid-evidence, feasible, nonterminal holding by one feasible bin, once per trip.
No replacement vehicle or retrospective selection by actual return/assignment.

Six arms: upper {0,-30,+30}s x lower {unchanged,reduce}. Keep 12 fixed vehicles,
seven bins, single shared upper coefficient [-60,+60], EMA .35 and native 900s
zero-sum planner/execution gates. No new slack, fleet, terminal authority or
service deletion; leave later holding feedback intact. Future labels are not inputs.

## Independent futures and joint acceptance

Reserve `2140000000+100*context+future`: select 0--15, evaluate 16--31;
smoke `2149000000+context`. Exact common anchor, frozen V68/V211 innovation fork.
Select passenger minimum under all nonharm mean constraints versus ID0 in pools
all6 / upper-only {0,2,4} / lower-only {0,1}; ties smaller ID, fallback0. Seal all
choices before evaluation. Budget: 4 references, <=6 smoke, 384 select, <=256
evaluation continuations; no adaptive extension.

Joint must beat bilateral025 AND EACH selected single layer: passenger95%CI
upper<0; actual full-service arrival CV2, readiness, TOTAL public-clock launch
delay and unlaunched/incomplete counts CI uppers<=0. Both crossed actor/day and
conditional-future bootstrap (5000,seed214), plus both fixed8-future halves, must
pass. Passenger metric is full-horizon waiting+onboard person-seconds; report
all-demand journey too. Delay uses ORIGINAL public clocks, including control wait.
Keep physical return/launch clocks alongside eligibility-based readiness.

Conserve vehicle elapsed=executed holding+other elapsed. No-op/blocked/changed
assignments stay in outcomes; unsupported labels stay missing. Uncleared service
blocks admission. Each evaluation half needs a real joint upper/lower execution;
supply-chain attribution also needs changed selected-trip return and earlier
same-terminal launch of that vehicle. These checks never filter outcomes.

## Audit and next step

Preregistration t211126 exited0 with8 tests; no replay at that stage. Later
high/manual t211134(source ee908c69b6,node004) passed14 tests and completed all
five fixed smoke paths. Archived endpoint,full passive/off action/timing/resource
equivalence,and identical joint repeat passed. The real-reduction gate FAILED;
exit1 is a support rejection,not a task to retry or expand.

Read-only high/manual t211136(source8b4311aa35) exited0 with2 tests; retrieved
only8984-byte JSON. At anchor7560s,bus9/trip31 has2 window decisions and
bus10/trip33 has6. ALL8 request0s,with valid forward/follower evidence,no mask
and no post-window queries. Both baseline and joint have0 registered changes.
Other incoming trips offer14 valid positive queries across5 trips;these do not
yet establish terminal-queue relevance or immediate vehicle-supply linkage.
Journey,passenger time,actual CV2 and total public-clock delay differences are0;
readiness mean changes-0.061068s/trip. This is not a joint-control improvement.

Engineering failures t211129--t211131 exposed the nested runtime-proof path,
missing native ready-clock field and old V211's exactly-two-decisions finish.
Fixes preserve native dynamics and record ready time passively,without backfill.
The cost amendment was sealed BEFORE t211134:3 prior full simulations plus
5 fixed reruns=8(original6 cap),with1 pre-action abort and1 pre-simulation failure
separate. Same ID3/future2149000000;no scientific budget/gate/choice changes.

Stop V214:selection/evaluation stay closed,no more replay or main-policy change.
Next needs a NEW causal event-based preregistration:require an actually observed
positive holding opportunity,then verify its current terminal-queue/supply link,
rather than locking the nearest two vehicles regardless of their action support.

Limitations:one smoke future,consumed prefix;not fresh-day policy confirmation,
online deployment,strong-fixed superiority or frequency-gain evidence.
