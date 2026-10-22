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

High/manual t211126 on node004 exited0: 8 tests, runtime interface audit and
known scheduler/protocol namespace checks passed. Retrieved only2026-byte JSON.
No replay,smoke,training,fitting or main-policy change. Next requires separate
authorization: passive capture/equivalence and nonzero-opportunity smoke first.

Limitations: consumed prefixes and conditional future innovations, not fresh-day
policy confirmation, online deployment, strong-fixed or frequency-gain evidence.

Smoke implementation ee908c69b6: high/manual on node004. Five paths:
original baseline, passive baseline, separate smoke-future baseline, fixed joint
ID3(-30s/reduce), identical joint repeat. Compare archived endpoint and full
actions/timing/resources; require real issued reductions and changed plan clocks.
Keep the native transition-mode last nonterminal holding opportunity. Selection
and evaluation banks stay closed; no matrix or training is authorized here.
Startup t211129 exited1 before simulation (nested runtime-proof path); t211130
reproduced the archived baseline, then stopped at the first passive query because
the frozen Bus lacks pre_action_time_s. The fix passively timestamps the native
observation-ready callback; no current-time backfill. Reuse the same five paths,
candidate and future. t211131 then completed a second native baseline and full
passive simulation,but V211's frozen finish rejected fewer than two old-selector
decisions. V214 now owns its physical finish and retains real0/1/2 support;
the nonzero executed-reduction smoke gate remains mandatory.

Engineering-only amendment before the next attempt:three prior full simulations,
one pre-action abort and one pre-simulation failure are retained. Cap total full
engineering simulations at8(original6);rerun only the SAME five paths with ID3
and2149000000. Seal this in engineering_recovery.json before replay. No added
candidate/future,selection/evaluation budget increase or scientific gate change.
