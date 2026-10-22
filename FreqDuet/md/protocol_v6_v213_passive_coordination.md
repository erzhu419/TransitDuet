# V213: passive reconstruction of consumed V211 selection paths

Reuse only the four V211 contexts,selection futures0--15 and all15 candidates:
960 existing paths. No evaluation futures,new demand days,selection updates,
training,controller change or interval fitting. Original12-vehicle fixed pool,
seven holding bins,shared upper[-60,60],EMA and zero-sum projection unchanged.

Use the frozen V211 source/runtime/checkpoints on the server. Add only passive
observation in a separate script/output root. Save every upper callback's
before/after pending-trip plans and fresh command;every actual launch and return;
lower query-time context,load,predecessor and executed holding-so-far;issued
holding and arrival/departure times. Completed ready-neighbour labels are offline
analysis only and never supplied to the policy. Store compressed traces remotely.

Before expansion,run synthetic tests and four paired passive-off/on replays
covering baseline,pure reduction and joint interventions. Require exact full
action events,arrivals/departures,launches/returns and outcomes. Every formal path
must reproduce ALL archived V211 fields,including common prefix,two interventions,
seven-bin counts,resource ledger and endpoints;on disagreement stop that path.
Old full trajectories were not saved:full-path observer equivalence is tested
in smoke,not claimed as a comparison against nonexistent historical traces.

Compare each lower pattern with lower0 at the SAME upper setting,and upper-only
with baseline. Split actual matched command differences into registered events,
after BOTH second-intervention times,and other events;unmatched identities remain
separate. Close the total against archived whole-day holding. Match actual launches
by trip and report time/vehicle changes. Report subsequent positive holding changes
by completed-neighbour mechanical marginal squared-gap correction,unresolved
support and onboard exposure proxy;retain passenger/CV2/readiness/public-clock
delay effects. These labels distinguish locally compensatory and counterproductive
responses,not the causal necessity of individual actions without extra interventions.
The next service of the SAME physical vehicle is not a follower label;retain it
as unsupported rather than treating the next round trip as a large following gap.

All tests,replays and analysis use scheduler high/manual on node004,where the
frozen archives reside. Parallel groups are allowed after smoke. Local retrieval
is limited to smoke/shard markers and final small JSON summary;no raw traces/ckpts.

## Execution

Smoke c00e43dcac/t211087 exited0:8 tests and four paired paths(eight rollouts)
passed exact full action/event,timing,dispatch/return and archived-endpoint checks.
Analysis-only follow-up f30f6691e7/t211089 exited0 with9 tests,including the
same-physical-vehicle unsupported label. Its passive recorder is unchanged.

High/manual formal tasks t211093--t211096 cover groups0--16/16--32/32--48/48--64,
16 workers each on node004(64 in total). Reuse the four completed passive paths;
all other paths must match their original selection records before saving.
Aggregate t211097 waits for all four small shard markers;it does not consume
partial output. Only markers and final summary sync locally;traces stay remote.
No controller admission or new scientific result is implied by the smoke pass.

## Outcome and next decision

All four replay shards exited0. Aggregate t211097 completed after the local
completion markers arrived;no replay recovery was needed. Read-only high/manual
reviews t211101(source3a4aab70f1,7 tests) and t211110(sourced61c5cdeec,8 tests)
also exited0. Three small JSONs total219668 bytes;399080491 bytes of compressed
trajectories remain on node004. No new replay,training or main-policy update.

All960 stored results,full query action histograms and registered commands match
V211. Actual arrival CV2,dispatch gaps and readiness reconstruct from the traces.
Each upper-only arm changes4 planned slots/path(256 instances),but0 corresponding
actual launches. Plans persist:these are not later-plan erasures. All256 assigned
vehicles and preceding return times match the control;launch is always return+1s.
Control eligibility-to-launch waits span73--703s(mean369.621s). Moving eligibility
alone changes this wait by-2.5/+2.5s per affected slot without moving its launch.
The two registered lower locations hit0 immediate inbound supplying trips/256
and only4 assigned physical vehicles/256 in each arm. V212's fixed-upper identity
checks align these locations across lower patterns. The pilot did not directly
coordinate holding on the vehicles supplying these blocked dispatches.

At upper0,pure(-1,-1) changes holding-3.906s at the registered events,+12.969s
after both interventions,and+9.063s over the day. Downstream gross increases
132.344s and decreases119.375s/path nearly cancel. Among1346 increased events,
completed-departure mechanical comparisons are positive996,negative331,zero4
and unresolved15;69 ready-stage-negative events become departure-positive.
CV2 still rises1.162e-5:holding variance contributes-1.377e-5,but its cross terms
with release/other elapsed contribute+3.487e-5. A blanket rebound suppression
would remove many locally compensatory actions,not just counterproductive ones.

Relative to bilateral025,means(journey seconds/passenger,CV2,readiness seconds/trip):
pure reduction(+.004819,+1.162e-5,+.039897);upper-30(-.076940,+2.838e-5,-.179091);
upper+30(+.004394,-2.330e-5,+.056059). No stable joint improvement is established.
For upper-30,the crossed actor/day journey interval remains[-.172759,+.013973]s.

Next:preregister a NEW inbound-resource-coupled pilot before any control replay.
Generate locations from current direction,position,nonzero holding opportunities,
terminal queue and executed holding;future return labels remain evaluation only.
Coordinate with actual vehicle supply,not only nominal timetable changes;retain
the original fleet,seven bins,upper limits and public-clock delay endpoint.
Use disjoint selection/evaluation futures after checking namespace consumption.
Do not extend the existing two-location search,freeze a supposed winner,or train
longer from this diagnostic. No next-stage control experiment was started.

Limitations:four consumed prefixes(two actors/two days),not independent policy
confirmation. Completed-neighbour scores are mechanical comparisons,not causal
necessity/harm labels. Old full trajectories were not archived:full passive/off
trajectory equivalence is established in four smoke pairs,not960 historical
event-by-event comparisons. Readiness is eligibility-relative,not pure vehicle
availability;retain the separate return ledger and public-clock launch delay.
