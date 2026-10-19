# V133: passive launch-origin ETA handoff

Four consumed paths: actors12011/933013, day2231004, original split lower with
terminal plan off/on. Actor, critic, dual, terminal rule, gains and bins freeze.
Shadow only; original predictor output still drives every actual decision.

Extend route/current-speed fallback only on the follower's first running segment,
with an already observed actual launch and spatially observed progress. At the
launch tick, that observed launch supplies origin0; all historical component
labels still finish strictly before prediction time. Journey-source predictions,
later segments, missing support and absent followers are unchanged.

Eight focused tests precede four parallel server rollouts. Require original
endpoints, arrays, executed actions and valid saved decision/evidence fields exact.
After rollout, label the same forecast vehicle/trip's ready time; report missing
identity outcomes separately. Compare original vs proposed MAE and hypothetical
holding only, not candidate passenger/CV/turnover effects.

Admission to a separate closed-loop pilot requires all four passive paths exact,
all modified forecasts resolved, pooled MAE no worse, and the consumed12011
trip16/stop5 shadow command matching the original no-terminal0 s command. This
small development diagnostic cannot promote a controller. No new training or
demand days. Scheduler CPU jobs on node001--006; local only tiny markers/summary.
