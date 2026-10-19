# V132: passive terminal/lower evidence replay

Replay only the consumed V130 split-plan paths on day2231004 (actors12011,
933013); reuse both V124 baseline traces. No training or controller changes.
Keep ready-time recorded AVL estimates, follower identity, online prediction, query-time
context and selected/executed holding separate. Never rerun a predictor for
logging. Before analysis require V130 endpoints, every timing array, terminal
commands/dispatches/returns and all original executed lower actions exact.

Focus on actor12011 trip18's +1 s terminal command and trip16/station5's 0->30 s
holding change: distinguish actual policy-query clocks, identity/source switches
and action thresholds. Contrast actor933013's first divergence. Four focused
instrumentation tests precede two independent replay jobs; aggregate on server
and fetch only the small summary. These are consumed-path mechanism diagnostics,
not new efficacy evidence or grounds to change the main policy.

## Outcome (2026-10-19)

Source cb062b9fd8: tests t195221 and replays t195222/node006, t195223/node005
exited 0; both endpoints, all timing arrays/terminal records and 10400 executed
lower actions are exact. First aggregation t195224 exited 0.
Final tests and filtered summary: t195225/node006 exited 0, four tests passed;
only the 50547-byte summary was fetched (plus tiny completion receipts).
Actor12011: trip16/stop5 arrives3231, ready3241, queries3242; follower stays
bus9/trip18. Its ETA source changes journey-speed -> route-speed, bypassing the
component model: request gap357.733 -> 635 s, desired hold0.967 -> 35.625 s,
quantized0 -> 30 s. This is a source-handoff discontinuity, not identity switching
or a query before the terminal command. Later 970 actions change, total -1270 s.
Actor933013: first changed hold is trip22/stop1, query4023 -> 4053, forward
gap279 -> 309 s, desired20.25 -> 12.75 s, quantized20 -> 15 s; later410 actions
change, total -145 s. Source loss first appears earlier but leaves action0.

Next: examine a causal, origin-consistent predictor handoff for just-launched
followers, first on consumed paths; keep gains/bins/terminal rule fixed and main
unchanged. Daily benefit is not identified by the first changed action alone.

Limitations: pinned runtime has no cached AVL input snapshots. The legacy
`ready_avl` empty/null fields are unavailable, not proof of no follower; final
summary excludes them. Existing identity/source/gap/query records remain valid.
