# V70: independent balanced-prefix label confirmation

Freeze main, the four V59 updated actors, upper/B200 deployment, dual and all
value models. No fitting or controller promotion. Use the unchanged V69 event
coupling, seven holding actions and exact completed-prefix contract.

Predeclare demand seeds 700001-700012, disjoint from V64-V69 development and
validation days. Four actors x 12 days x all three targets (7200/21600/36000s):
48 full-day references, 144 label-independent first-eligible prefixes.
Private pre-prefix sampler seed: `7000000 + case*1000 + day_index`.
Each prefix uses 32 new futures, seed `70000000 + context*100 + future`;
seven arms share that future and event uniforms. Total: 4,608 grids / 32,256
action replays, plus 48 references. Every day contains all three windows.

Select on futures 0-15 and evaluate on 16-31, then reverse and average.
Selection minimizes passenger exposure subject to mean CV-squared, readiness,
unlaunched and incomplete trips no worse than native; ties preserve native.
Only held-out halves enter the primary selection effect. No threshold tuning
or action reselection using evaluation outcomes.

Report signed noise-corrected signal squared, single-future and 32-mean SNR,
16/16 sign agreement, held-future passenger/constraint deltas, and descriptive
window/training-seed/day breakdowns. Bootstrap 12 demand-day clusters after
averaging actors and windows within day. Passenger unit: full-day person-seconds.
Independent-label gate: passenger and CV-squared corrected-signal CI lower >0
and 32-mean SNR CI lower >1. Joint-opportunity gate: passenger delta CI upper <0
and all four constraint delta CI uppers <=0. Report both gates separately;
neither changes deployment automatically.

Server smoke: affected V67-V69 tests, V70 tests, one cached V69 native endpoint
identity, two new references, three complete grids spanning all windows/two
original training seeds, and three forced-native full-path identities.
27 smoke replays; 23 reused formally. References/matrix/aggregation are gated
through scheduler on node001-node006 only. Pull small JSON/markers/logs only.

Limitations: fresh prefixes and futures, but the same four actors from two
original training seeds; not new-training-seed or deployable-controller
confirmation. Pending arrivals and realized day profiles stay conditional.
Per-prefix selection uses counterfactual labels, not an online observable scorer.

## Execution

Source `b3259d1997`, immutable snapshot `source/v70_independent_20261016`.
Smoke `t165690` passed on node004: 31 tests and all 27 replay checks.
References `t165693`: 48 trajectories, 24 workers; two smoke references reused.
Matrix `t165694-t165697`: 4,608 grids, planned workers 32/80/80/80 on
node002/004/005/006, gated on complete reference export; three smoke grids reused.
Aggregate `t165698` depends on the four matrix completion markers. Scientific
results remain pending; only the 104-byte smoke marker was pulled locally.
