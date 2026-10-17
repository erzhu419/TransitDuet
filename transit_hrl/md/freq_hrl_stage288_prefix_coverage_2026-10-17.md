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
Continuations `t169842`-`t172961` all completed with exit0: exactly 3,120 distinct
tasks, zero missing/extra signatures, 87,360 qualified native continuations.
Node002/004/005/006 completed 781/787/776/776 tasks; no node pin or protocol
change. Fit `t173070` (node002) and report `t173071` (node006) completed with
exit0. All 96 residual fits and held choices were frozen before held labels
were opened; zero raw-reference fits/actor updates. Maximum native RSS was
701.40 MiB. A reporting-only addition measures training control outcomes on
the saved choices; its synthetic round-trip test passes. Only 138,071 bytes
of fit/scalar reports were pulled this turn, not models or trajectories.

## Results and decision

Both fresh refits improve held cost versus raw, but neither preregistered gate
passes: widening coverage does not resolve fleet error or joint cost/wait.
All feasible-margin MSE entries below are x1e6, ordered cost / fleet / service.

| Evaluation | Dense | Wide |
| --- | ---: | ---: |
| Fitted 16 futures | 653.943 / 650.030 / 68.363 | 655.534 / 651.194 / 67.994 |
| Independent 8 futures | 1179.575 / 1197.117 / 91.266 | 1182.125 / 1198.808 / 90.850 |
| Independent 8, minus MC variance | 453.850 / 466.625 / 70.454 | 456.399 / 468.315 / 70.038 |
| New-prefix 16 futures | 570.696 / 651.443 / 77.551 | 569.739 / 651.439 / 77.077 |
| New-prefix 16, minus MC variance | 251.497 / 327.566 / 66.301 | 250.540 / 327.561 / 65.828 |

Wide minus dense held fleet MSE is -0.00489 [-1.46998, 1.44078] x1e6:
only a 0.0015% point reduction in noise-adjusted error. Held cost/service MSE
differences also cross zero. Against unchanged247, wide improves fleet MSE
(-6.979 [-10.908, -3.636] x1e6), but worsens service MSE
(+3.177 [1.797, 4.630] x1e6). Against raw, both refits improve cost/fleet
prediction while significantly worsening service prediction.

Cost below is x1e6; wait is minutes; intervals are the frozen 40-prefix CI95.

| Comparison | Cost [CI95] | Wait [CI95] |
| --- | ---: | ---: |
| Stage247 - raw | -126.63 [-220.57, -41.14] | -0.0001412 [-0.0004554, 0.0001277] |
| Dense - raw | -195.18 [-323.94, -81.12] | 0.0002597 [-0.0000921, 0.0007029] |
| Wide - raw | -218.29 [-342.88, -103.51] | 0.0002220 [-0.0001221, 0.0006419] |
| Wide - dense | -23.11 [-74.56, 26.06] | -0.0000377 [-0.0001277, 0.0000486] |
| Wide - Stage247 | -91.65 [-204.09, 14.22] | 0.0003632 [-0.0000434, 0.0008124] |

- Training control points (cost x1e6 / wait minutes versus raw): dense
  fit16 -343.48 / -0.0003921, independent8 -185.24 / -0.0003054; wide
  fit16 -320.02 / -0.0004390, independent8 -156.68 / -0.0002984. Independent
  futures retain only about half the fitted cost benefit, mainly losing fleet
  savings. These are descriptive points, not additional CI-supported wins.
- On held prefixes, wide fleet -236.45 offsets service +18.17, giving net
  cost -218.29. Cost improves in 26/40 prefixes and in both root/five-regime
  means. This is genuine conditional cost evidence, not evidence that wider
  coverage caused the gain: dense also improves cost, and wide-dense is mixed
  across roots. One prefix supplies about 87.6% of that comparison's net cost
  point benefit. Subgroup/concentration figures are descriptive.
- Independent8 raw MSE includes larger label-mean variance; noise-adjusted
  held errors are actually
  lower. The observed problem is the service/wait tradeoff and the absence of
  a wide-over-dense effect, not uniformly failed held prediction.

Decision: do not promote wide or add more coverage sampling in this fixed
class. Retain the new-data refits as conditional cost baselines. Before another
learned selector, test whether waiting-constrained joint action gains survive
independent futures in the new training bank; reserve new prefixes for any
subsequent confirmation. No further fits/simulations were dispatched.

Limits: conditional on fixed historical representations and rotations; single
intervention with frozen lower, not an end-to-end or training-seed HRL claim.
Coverage also changes fitted precision/shrinkage. MC-variance subtraction is
an independent-label error estimate, not an unbiased in-sample correction;
different prefix populations need not have equal intrinsic difficulty.
Scheduler wall timestamps for the short report `t169732` imply a negative
duration (-14.02 s); they are not used for runtime accounting. Worker receipts
carry monotonic `elapsed_s`.
