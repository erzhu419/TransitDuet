# Stage288: matched training-prefix coverage

Stage287 stops the complex level-only branch. Test independent training-scene
coverage in the fixed simple Stage247 class, not another head/loss variant.

- Fresh training: 60 prefixes (six per root/regime), all six probes. Six fixed
  rotations: dense uses one prefix per stratum and six probes; wide uses all
  six prefixes per stratum and one Latin-rotated probe each. Both fit exactly
  60 states x 16 futures. Probe/root/regime counts match. Across rotations both
  arms consume the same 360 states exactly once; rotations are not seed replicas.
- Freeze original raw reference, original training TSVD whitening, selected
  rank/alpha, A-RMS ball, scale and cached head difference. Refit only the
  state-GLS head sum. Keep the eight zero profiles zero. Precision/shrinkage
  use only that fit's assigned labels under the existing recipe; coverage also
  changes the estimator's group calibration, not just regression geometry.
- Another eight futures per training state measure independent training-state
  error. Forty fresh held prefixes (four per root/regime), all six probes and
  16 futures, measure transfer. Freeze every model/held action before reading
  held labels. Stage287 confirmation labels and all software-pilot labels are
  excluded from fitting and selection.
- Compare feasible-margin cost/fleet/service MSE on fitted16, independent8 and
  held16; report independent label-mean variance separately. Pair held errors
  and control outcomes before a stratified 40-prefix bootstrap (5000, seed288).
  Paired conditional control MC standard errors are reported separately.
  Fleet-error improvement and cost/wait improvement are separate conclusions.
  Control advancement requires wide to improve cost and wait versus dense,
  raw and unchanged Stage247. No best rotation/profile selection.
- All six actions use the existing strict native prefix/state/RNG and common
  future replay. Main budget: 100 source episodes plus 87,360 continuations,
  3,120 chunks, 96 small residual solves; zero raw-reference fits/actor updates.
  First qualify ten separate cross-worker native pilots and solver startup
  (150 episodes including captures). Main collection waits for this gate.
- One CPU per task, 1.5 GiB native / 3 GiB offline, any node001-node006,
  no node pin. Only small receipts/scalar reports are pulled.

Run: `native_transit_prefix_coverage_stage288_20261017_r1`.

Execution: eight focused tests passed, including a toy fit/report round trip
and phase-gate regression; both CLI entry points start. Freeze `t169665`
qualified all 16 profiles (eight active), maximum reconstruction error
`1.67e-16`. Pilot captures `t169667`-`t169676` and paired replays
`t169679`-`t169688` all completed with exit0. Report `t169732` qualified 150
native episodes and eight constrained fits; maximum KKT `8.85e-15`, native RSS
681.16 MiB. No pilot outcomes were used for selection. Main source captures
`t169737`-`t169836` also completed with exit0 (100 prefixes, six states each).
Continuations `t169842`-`t172961` are registered: exactly 3,120 distinct tasks,
zero missing/extra signatures, 87,360 planned native continuations. No node pin
or protocol change. Registered-task input staging was completed before retrying
dispatch. The first sampled 32 train/held workers launched eight per node on
node002/004/005/006; continuation workers `t169842` (train) and `t172002` (held) were
verified running. The first train probe8/draw0 six-action replay qualified
under the exact prefix/state/RNG checks. No main performance conclusion yet.
Only the 2,839-byte pilot summary was pulled this turn.

Limits: conditional on fixed historical representations and rotations; single
intervention with frozen lower, not an end-to-end or training-seed HRL claim.
Scheduler wall timestamps for the short report `t169732` imply a negative
duration (-14.02 s); they are not used for runtime accounting. Worker receipts
carry monotonic `elapsed_s`.
