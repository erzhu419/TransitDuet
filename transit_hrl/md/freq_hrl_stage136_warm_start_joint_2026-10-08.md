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

## Run Receipt

Implementation `273f6ba6c5`, registration `1d80ae9c80`; nine related tests passed,
including a reduced native-path runner and exact joint/separate parameter
identity. Both completed source cells and all four selected upper files are
available on node004; no checkpoint was downloaded. Run
`pointmaze_warm_start_joint_stage136_probe_20261008_r1`: `t136900` root410037 and
`t136901` root410049 accepted. Both are DONE on node001/node005.65,540bytes
of scalar JSON were fetched; no checkpoint or raw trajectory was downloaded.

## Result And Next Step

Both roots satisfy the native/optimizer budget and exact layer isolation:
960episodes/1,152,000steps. Equal-root descriptive means are:

| Period | Warm start - forecast | Upper update - warm start | Lower update - warm start | Joint - warm start | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 1.607816 | -3.175088 | -0.051047 | -2.624583 | 0.601552 |
| 100 | 0.821580 | -2.782663 | -0.373809 | -2.382020 | 0.774452 |

Upper-only and joint updates lose at every root-period. The interaction is
positive, so harmful coupling is not the primary observed failure. Ordinary
upper PPO increases mean-output RMS5.85-8.76times. Its critic advantage variance
is90.7%-95.3% temporal; independent-noise upper score cosines are mostly negative.
LOO centering does not repair long-period agreement (-0.225279/-0.197175).
Upper sampling itself costs2.116522/4.162028 return relative to the mean policy.
These observations identify scale/credit concerns, not a proven repair.

Stage137 reconstructs this exact bad upper update from its recorded training
paths, then compares fixed-KL rescaling and projection into the existing26D
subspace. Both signs are retained on new evaluation scenes with lower frozen;
no evaluation-based choice or seed extension. The critic/exploration mismatch
remains open. Stage135's frozen-lower gain and Stage121's failure stay unchanged.

## Limitations

This does not fix the critic or guarantee joint PPO improves. It tests a warm
start and update decomposition, not learned promotion, frequency responsibility,
domain generalization or top-conference readiness. Prior negative results stay.
