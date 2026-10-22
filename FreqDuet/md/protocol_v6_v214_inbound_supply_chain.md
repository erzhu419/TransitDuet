# V214: inbound supply-chain joint counterfactual preregistration

Freeze the conditional opportunity pilot below, not a new main policy. This
stage authorizes metadata/interface audits and synthetic tests only; no replay,
smoke, training, interval fitting or candidate selection is started.

## Selection and authority

Use V211's four consumed B200/bilateral025 actor/day prefixes. At the first
fresh native upper decision with actual query time in [7200,8100), require a
currently due same-direction terminal queue and no compatible ready vehicle.
Rank opposite-direction active vehicles by current remaining route distance,
then bus ID and trip ID; commit the first two (one if only one exists). At least
one must have a native nonterminal controlled stop at/ahead. Do not skip a nearer
vehicle because a farther vehicle later turns out to offer a better action.
No anchor in any context stops the pilot, rather than replacing the context.

At each committed physical trip, select its first subsequent nonzero bilateral
holding query with valid forward/follower evidence and a smaller feasible bin,
before anchor+900s. Reduce by one adjacent feasible bin, at most once per trip.
Keep zero opportunities, no replacement vehicle, no retrospective matching to
the vehicle that actually supplies a launch. Controls log shadow opportunities.
Positions, current plans/queues, action masks and evidence are query-time inputs;
future return times, realized following gaps and future assignment are labels.

Six arms = upper coefficient shift {0,-30,+30}s x lower {unchanged,reduce}.
Retain exactly 12 fixed-pool vehicles, seven bins {0,5,10,15,20,30,45}s and the
single shared antisymmetric upper coefficient clipped to [-60,+60]. Preserve
native EMA .35, 900s rolling-zero-sum planning and execution gates. Never issue
slack credit, enlarge the fleet, delete service or directly override launches.
Both directions and all downstream holding feedback remain in the outcomes.

## Independent futures and joint acceptance

Reserve `2140000000+100*context+future`: selection 0--15, evaluation 16--31;
separate smoke `2149000000+context`. Fork innovations only after an identical
pre-command anchor, preserving the frozen V68/V211 coupling and current latent
window/day. Maximum four references, six smoke replays, 384 selection and 256
evaluation continuations. No adaptive extension. Seal choices before evaluation.

Selection pools: all six / upper-only IDs {0,2,4} / lower-only IDs {0,1}.
Choose minimum mean passenger person-seconds subject to all constraint mean
deltas <=0 versus ID0; ties use smaller ID, ID0 is fallback. Also seal best-single.

Primary contrasts are joint versus bilateral025 and EACH selected single layer.
For every contrast require passenger-time 95% CI upper <0; actual arrival CV2,
readiness delay, total public-clock launch delay, unlaunched and incomplete-trip
CI uppers <=0. Both crossed actor/day and within-context future bootstrap
(5000 draws, seed214) must pass. Each fixed eight-future evaluation half must
also have passenger improvement and nonharmful constraint means. No tradeoff
score or relaxed margin. Report selection versus held-future effects separately.

Passenger time is full-horizon waiting+onboard exposure, with all-generated-demand
journey time reported alongside. CV2 uses frozen full-service actual arrival gaps,
not dispatch/local CV. Readiness keeps its original eligibility clock; separately
track actual inbound return and launch. Total launch delay is the SUM of positive
actual-minus-ORIGINAL-public launch times, including control waiting; report its
mean/counts too. Missing/unlaunched service is not a zero-delay observation.

Conserve each vehicle trip's elapsed=executed holding+other elapsed. Keep every
zero opportunity, clipped upper action, blocked launch and changed assignment.
Require paired outcomes and authority/conservation checks. In each evaluation
half, observe a joint path with changed upper command/plan AND nonzero lower
reduction; supply-chain attribution additionally requires an altered selected-trip
return followed by earlier same-terminal launch of that physical vehicle.
These are admission checks, never filters for the outcome population.
Any uncleared arm blocks admission; retain censored outcomes without inventing
launch times or zero delay for missing service.

Next, ONLY after separate authorization: implement passive query-time capture
and smoke exact action/timing/endpoints, then the bounded pilot. The executable
registration is `preregister_freqduet_v214_inbound_chain.py`; it contains no replay
entrypoint. Metadata audit/tests run via scheduler high/manual on node004; only
the small JSON summary returns locally.

Limitations: this is four consumed prefixes with independent future innovations,
not a new-day policy confirmation or deployment-valid observable-state planner.
It cannot establish superiority to strong fixed or frequency-separation gains.
