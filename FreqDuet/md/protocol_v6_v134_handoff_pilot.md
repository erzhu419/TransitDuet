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
