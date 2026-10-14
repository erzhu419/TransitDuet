# V34: Paired checkpoint trajectory diagnostic

V9 at 200 episodes retained a favorable journey effect but failed the registered
headway-CV magnitude and interval gates. V33's unfrozen control improved on its
development roster, while lower-actor freezing did not repair the effect. V9
and V33 also used different source snapshots, so their endpoints alone cannot
identify a learning-duration effect.

This diagnostic fixes the model to V9 commit
`673355cae10640b2737b6d51d541a9685059c342`. It trains the original reference,
compact-context control, and current policy on the eight V9 and four V33
training seeds. Every policy is evaluated on all eight V9 and four V33 scenario
seeds at episodes 39, 79, 119, 159, and 199: 36 training runs, 2,160 frozen
rollouts. These already-used seeds define development evidence.

Only checkpoint saving and trip-dump settings change. Training completes before
frozen evaluation starts. A server smoke compares final actor, critic, and
deployment state after two identical trainings with and without intermediate
saves, and verifies loading and evaluating both saved checkpoints.

The analysis reports each checkpoint's paired current-minus-control effects,
then the paired change in that effect from episode 39 to 199. Crossed-bootstrap
intervals reuse the existing estimator. Training/evaluation cohorts are also
reported separately. Headway CV, restricted journey/wait/in-vehicle time,
holding, denied-trip rate, readiness delay, upper/lower action means and spreads,
and regularity-evidence coverage are retained. Action summaries are descriptive.

The 200-episode endpoints on the original V9 seeds are checked against the
archived evaluation rows. A mismatch requires a reproduction diagnosis before
attributing a difference to training duration. No checkpoint is selected as a
new paper main, no confirmation gate is relaxed, and no V33 reserved seeds are
used. A later independent confirmation needs a separately frozen design.

All training, smoke, and aggregation run through scheduleurm on node001-node006.
Model snapshots and checkpoints stay on the server. Only the small summary
CSV/JSON files are eligible for local retrieval.

Server smoke `t151969` passed five tests, but its equivalence comparison
incorrectly required configuration metadata to match across different save
frequencies and logging paths. The failure left the formal matrix waiting for
its success marker. Fix `0a5542347a` validates each saved configuration against
its runtime configuration and omits cached diagnostic timing, then compares all
operational deployment state and complete actor/critic checkpoints. Inspection
`t152019` found only configuration identifiers and wall-clock timing differed.
Recovery `t152020` reused the saved models: eight regression tests, all three
checkpoint comparisons, and both frozen evaluations passed. Its success marker
was synchronized and the obsolete V34 failure escalations were resolved.

Dispatch: diagnostic source `0a5542347a`; recovery/release task `t152020`,
36 trajectories `t151972-t152007`, and aggregation `t152008`. Every task is
restricted to node001-node006. All 36 trajectories are running: six each on
node001/node002 and eight each on node004/node005/node006. Node003 was full, so
the scheduler used other allowed nodes. Only aggregation awaits completion.
There is no new efficacy result or change to the paper controller yet.

## Runtime optimization (2026-10-14)

Server profile `t152418` found 912,985 keyed random-generator constructions,
accounting for 71% of profiled rollout time. Reusing a scalar generator with
the same legacy seed preserves keyed draws; persistent streams stay separate.
Six paired full rollouts cover all three configurations and both evaluation
cohorts: all 390 non-timing/non-run-identifier columns and complete lower/upper
action/reward traces match exactly. Main/45007 takes 169.5 s originally versus
41.4 s optimized (4.09x; aggregate 4.14x over six pairs). Tests `t152425-t152426`
pass 16 cases; `t152433`
matches both episodes' lower, upper, and operational deployment checkpoints
against the archived original smoke.

The immutable V9 snapshot and 36 running training workers remain unchanged.
Tasks `t152434-t152541` precompute the already-available episodes 39/79/119 with
the original V9 evaluator on five CPU nodes; owning workers reuse those files.
Optimized source is isolated in `FreqDuet-v6-engineering-v9-runtime-optimized`.
Next: finish the original V34 matrix and paired aggregation; use the verified
runtime optimization for subsequent experiments. Models and profiles stay on
the server; only small benchmark/training summary JSON files are retrieved.

At the user's request, the remaining 35 original workers and old aggregation
were cancelled. Audit `t152546` confirms all 36 trainings completed 200 episodes;
141 checkpoint evaluations are complete and 39 remain. Replacement evaluations
`t152549-t152587` use the verified optimized snapshot, retaining all training
checkpoints and completed evaluations. Two completion tests pass (`t152548`).
Aggregation `t152588` waits for the 39 replacement markers and preserves the
original analysis and historical-endpoint check. Runtime provenance is recorded
separately; no training is restarted and no experiment protocol is changed.
