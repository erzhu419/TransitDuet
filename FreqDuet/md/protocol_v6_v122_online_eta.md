# V122: ready-time online prefix equivalence

Keep the V120/V121 rule, 3600 s window, moving eligibility and raw fallback.
The live predictor receives only physical launch, arrival, ready and departure
callbacks plus current AVL evidence. Running labels mature at arrival, service
at ready, holding/ticks at departure; queries still exclude all equal-time labels.
Issue and cache the forecast at ready, not the later policy-action tick.

Replay the 16 consumed V121 actor/day paths, with no new demand days or training.
The first path of each actor must pass before the other 14 run on eight workers.
After each live replay ends, open the cached V121 artifacts: require unchanged
endpoints/accounts, every original decision and evidence row, all timing/launch
arrays, component eligibility/support and forecast values (absolute tolerance
1e-8 s). Every frozen negative/quota restoration sign must be identical.
The online predictor itself never receives cached future arrays or labels.

Keep raw online forecasts and replay arrays on the server. Scheduler on
node001--node006 runs all tests/replays; only final small JSON/logs return.
Quota, observations, actors, critic, dual and main stay unchanged. Passing this
stage establishes online implementation equivalence, not new prediction evidence
or closed-loop passenger/CV2 benefit. The next stage is a small matched control
pilot only after this gate, with journey, actual CV2 and readiness as outcomes.
