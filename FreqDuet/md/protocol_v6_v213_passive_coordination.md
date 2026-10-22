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
