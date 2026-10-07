# Stage129: Frozen Compact Upper Replication

Stage128 has positive gains in all four cells, but period100 mean+0.372044
does not clear0.5 and two development roots do not confirm the method.
Freeze its26-dimensional causal projection, native coordinate epsilon0.005,
unit damping, original-state Fisher radius0.000555556, actor variance, decoder,
0.05 response authority, lower and training-only quadratic step calibration.

Use six source-policy roots not used in Stage123-128 development:
410037/410049/410061/410073/410089/410101. Each receives four fresh label
scenes/two complete noise trajectories at every decision,16 separate calibration
scenes/two panels, and64 new evaluation scenes. Raw gets the same labels and
native calibration budget; flat/opposite/blinded controls remain. Both periods
50/100 are mandatory. No return-based root filtering or new hyperparameters.

Four primary endpoints: compact-minus-forecast and compact-minus-raw at each
period. Equal-weight source-root means;65,536 bootstrap draws, Bonferroni4,
familywise alpha0.05. Material replication requires forecast CI lower>0.5
and raw CI lower>0 at both periods. Report positive-but-subthreshold evidence
separately. Development roots are excluded from these intervals.

Per root:4,896label+320training+192crossfit+768evaluation=6,176episodes/
7,411,200steps. Six tasks,16workers+parent/12GiB each, dynamic node001-node006.
No waiting aggregation task: aggregate small JSON after the workers finish.
Four final upper weights/root remain server-side; no raw states/traces pulled.

## Run Receipt

Implementation `75ec71ca4c`, registration `e05721e12f`; eight focused tests
passed. All six source/forecast/lower artifact sets were read on node004.
Run `pointmaze_compact_replication_stage129_frozen_20261007_r1`:
`t136469-t136474` accepted, ordered by the six roots above. Task specifications,
seed roles, budgets and scheduler receipt are saved under this run directory.

## Completed Result (2026-10-08)

All six tasks completed;37,056episodes/44,467,200steps. Only178,968bytes of
compact JSON were fetched. All12 compact-minus-forecast and compact-minus-raw
root/period means are positive; all24 compact calibration crossfits are positive.
Root-cluster bootstrap CIs use the four-endpoint Bonferroni correction:

| Period | Compact minus forecast | Compact minus raw |
|---|---|---|
|50|+0.869393 [0.721655,1.014687]|+0.647773 [0.491357,0.776041]|
|100|+0.314953 [0.168861,0.440607]|+0.258889 [0.159622,0.340867]|

Positive incremental upper value replicates at both periods. Material gain0.5
closes only at50; at100 even the CI upper is below0.5. More seeds are not the
main remedy. Five of six period100 calibration fits select the unit-step cap,
with positive fitted derivative there; this motivates testing a second local
policy update, not proving that it will help. Return to the two development
roots, refresh native labels at the learned policy, and compare against an
equal-radius continuation of the old direction. Keep this replication intact.

## Limitations

Source policies were used in earlier research; these are not untouched external
benchmarks. This is frozen-method replication with fresh labels and evaluation,
not joint-HRL, learned promotion or frequency-specific confirmation. Stage128's
negative fold and gradient sign disagreement remain part of the evidence.
