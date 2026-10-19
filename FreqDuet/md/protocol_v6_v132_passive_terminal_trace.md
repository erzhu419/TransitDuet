# V132: passive terminal/lower evidence replay

Replay only the consumed V130 split-plan paths on day2231004 (actors12011,
933013); reuse both V124 baseline traces. No training or controller changes.
Keep ready-time frozen AVL, follower identity, online prediction, query-time
context and selected/executed holding separate. Never rerun a predictor for
logging. Before analysis require V130 endpoints, every timing array, terminal
commands/dispatches/returns and all original executed lower actions exact.

Focus on actor12011 trip18's +1 s terminal command and trip16/station5's 0->30 s
holding change: distinguish actual policy-query clocks, identity/source switches
and action thresholds. Contrast actor933013's first divergence. Four focused
instrumentation tests precede two independent replay jobs; aggregate on server
and fetch only the small summary. These are consumed-path mechanism diagnostics,
not new efficacy evidence or grounds to change the main policy.
