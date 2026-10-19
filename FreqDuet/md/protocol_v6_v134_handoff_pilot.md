# V134: small closed-loop launch-handoff pilot

V133 passed its frozen shadow admission. Use the same two actors/day2231004 and
terminal plan off/on: four candidate rollouts, four exactly reproduced V133
controls reused. This is a consumed development pilot, not confirmation.

Only deploy the already frozen V133 launch-handoff forecast. Keep weights, dual,
gains, bins, terminal rule and sampler seeds unchanged. Do not calibrate away
V133's remaining underprediction. Ordinary predictions change only through
endogenous state/history changes, not another forecasting formula.

Three tests precede four parallel scheduler/server rollouts. Report each actor's
paired passenger time, actual arrival CV2, total published-clock-to-launch delay,
native readiness, holding and completion. Include terminal x handoff interaction
and direct command changes on the same live states. A joint cell requires lower
passenger time and no increase in CV2, total launch delay or unfinished service;
show failed cells instead of masking them with pooled means. No population CI
from one day/two actors. Main remains frozen; independent-day evaluation requires
interpreting this pilot first. Local only small summaries/markers, no trajectories.

Outcome: source366cf193a8, tasks t195229--t195231/node006 exited0; three tests
and four parallel candidate paths completed. Three of four cells jointly improve.
Without terminal coordination both actors pass: mean journey-0.0801355 min,
CV2-0.006311535, total launch delay-5.089695 s/trip, holding-3.282443 s/trip.
With coordination12011 fails strict no-harm: journey-0.000798 min but
CV2+0.0000959354 and total launch delay+0.015267 s/trip;933013 jointly improves.
These are same-day development contrasts, not statistically confirmed harm/gain.
Next: freeze this forecast candidate and validate the no-terminal arm on new
demand days/remaining actors; no gain/bias/terminal-rule tuning or main switch.
Only the23113-byte summary and tiny receipts were fetched; V133 summary21869 bytes.
