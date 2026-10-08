# Stage136: Warm-Start Joint Update Diagnosis

Stage135 supports a small frozen-lower upper gain, not jointly learned HRL.
Stage121 joint PPO was harmful; Stage122 found temporal critic error but did not
validate an alternative actor direction. This pilot isolates the first PPO
update from the now-validated upper instead of extending selector training.

Use the first two Stage135 roots410037/410049, not the best-performing roots.
Load their validation-selected upper weights with exact saved-fit provenance;
keep the strong teacher, forecast, control authority0.05 and noise unchanged.
Periods50/100, horizon1200, eight fresh scenarios with two independent noise
folds. Sample upper with the original std0.15 for collection. Paired mean-upper
and forecast runs on those training scenes measure exploration cost separately.

From the same initial policy and16 collected paths, execute one update of upper
only, lower only, and both, using the shared SMDP-PPO actor/critic implementation.
Per-level optimizer seeds match across interventions. Joint upper/critic must
equal upper-only, and joint lower/critic must equal lower-only exactly; the native
performance interaction therefore cannot arise from different SGD shuffles.
MC-minus-original-critic remains the training signal. Independent-noise score
agreement, temporal variance and LOO are diagnostics only, not a new loss.

Evaluate32 separate paired scenarios on forecast, unchanged warm start, upper-
only, lower-only, joint and joint-blinded. Keep every effect and the interaction
`joint - upper_only - lower_only + warm_start`. Final single update only; no
evaluation selection, extra rounds or automatic seed extension. Two roots give
descriptive diagnosis, not confirmation CIs. Inspect which update loses gain
before redesigning credit or committing to a larger joint training run.

New cost/root480episodes/576,000steps; both roots960episodes/1,152,000steps.
Eight workers+parent/8GiB, scheduler dynamic node001-node006. No checkpoints or
trace writes; only scalar JSON is pulled. Stage135 and immediate earlier source
costs remain separate, with the rest of the source chain inherited.

## Limitations

This does not fix the critic or guarantee joint PPO improves. It tests a warm
start and update decomposition, not learned promotion, frequency responsibility,
domain generalization or top-conference readiness. Prior negative results stay.
