# V67: matched seven-action full-day causal labels

Freeze the four V59 post-64 actors, upper/B200 deployment, dual and all value
models. Eight new development demand days (670001-670008), private sampler
`6700000 + case_index*1000 + day_index`. Capture first eligible full-alphabet
decision at/after 7200/21600/36000 seconds: 32 native days, 96 prefixes.
Eligibility uses current departure evidence and action permissions, not labels
or critic scores. Keep all seven actions (0/5/10/15/20/30/45 seconds).

Replay each action from the same initial deployment through the full day:
672 intervention branches, 704 formal rollouts. Always draw the original
actor action before overriding it once. Check exact observable state, pending
commands, completed CV prefix, exposure, full lower/upper prefix, runtime RNG,
actor private sampling state, upper sampling state and scenario stream state.
Scenario tape seed/version stay fixed. Subsequent closed-loop actions and their
decision ordering may diverge; shared random innovations do not fix actions.
Every forced-native-action branch must match the full native holding/upper
paths and day outcomes. Reject projected actions instead of relabeling them.

Before the matrix: server tests, native reproduction against V60, and the first
complete seven-action grid/no-op smoke (nine rollouts, eight reused formally).
All computation uses scheduler on node001-node006. Prefix/RNG bundles, observable
features and action labels stay remote; only logs/markers/compact JSON come back.

Labels: terminal CV-squared, full-horizon passenger-person-seconds, journey,
holding, readiness/backlog and incomplete/unlaunched trips. Same-prefix
differences are the signed remaining CV-cost differences and once-only system
passenger reward differences. Store explicit observable features separately
from privileged replay/RNG audit fields; no learned action scorer is deployed.

Offline probe frozen V59, V65 linear/MLP and V66 shrunk action-cost differences
against zero-difference prediction. Exclude the reference zero action from
error averages; report non-tied cost pair ordering separately. Bootstrap eight
demand-day clusters after averaging actors/windows. Hindsight joint passenger,
CV and readiness opportunities are descriptive, not policy gains. No fitting,
model promotion or controller confirmation. These development labels cannot
serve as held-out evidence for a subsequently trained action-effect model.

## Execution

Source `5dd43f49d9`, remote snapshot `source/v67_labels_20261016`.
Smoke `t165468`: 45 tests passed; native V60 and complete seven-action/no-op
replays passed. References `t165475-t165479`: 32 native rollouts completed.
Interventions `t165481-t165485`: 337 planned workers over two waves on
node001/002/004/005/006; all 672 branches and aggregate `t165486` completed.
Actor, dual, value and main remain frozen; full labels stay on the server.

## Outcome And Next Step

All 96 prefixes matched and all 96 native-action branches reproduced the full
day; conservation error <=4.66e-10. Zero-delta CV-cost MAE was 0.00103042,
versus MLP 0.00129235 and best critic (shrunk linear) 0.00106672. Every critic
had worse MAE/RMSE with positive demand-day paired CIs. The V66 return-level
prediction gain therefore does not establish usable action-effect prediction.

There were 38/96 hindsight passenger opportunities under the CV/readiness/trip
constraints. Audit `t165490`: all eight days had opportunities; top one/five
prefixes contributed 10.5%/41.8%. These are development labels with one realized
continuation per prefix, not expected controller gains. Next: repeated matched
continuations to separate action signal from continuation noise, then fit
passenger/CV action-increment models and confirm on fresh days. Keep the actor,
dual and main frozen.
