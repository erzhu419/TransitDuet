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

Outcome: source aa998de9a3, tasks t195226--t195228/node006 exited0. Eight tests
passed; four parallel replays preserved all20800 decisions/evidence, arrays and
endpoints. All8 modified forecasts resolved; local ready-gap MAE475.545 ->83.569 s.
Anchor gap635 ->357.889 s, shadow hold30 ->0 s (baseline357.733 s/hold0).
The actual anchor gap430 s still exceeds the proposed357.889 s; these eight
observations share contexts and are not independent-day evidence. Admission
passed for a separate four-path V134 control pilot, not main promotion. Only the
21869-byte summary and tiny receipts were fetched.
