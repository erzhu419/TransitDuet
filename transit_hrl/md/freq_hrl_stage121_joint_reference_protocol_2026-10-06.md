# Stage121: Joint PPO Reference Tracking

Stage120 remains `inconclusive_no_automatic_seed_extension`: reference authority
improved, but period50 did not reach the registered conditional gain of 0.5.
Stage121 is a new joint-control development experiment, not confirmation of that gate.

Version2 fixes float32 GAE trace accumulation under NumPy2. The aborted version1
full run and its automatic retries are excluded. Full-run seeds, budgets, actors,
objectives and statistical thresholds are unchanged; native preflight now also
uses1200steps to exercise the actual long-episode path. The0.002 return-identity
check is unchanged. A regression reproduces the old error before the repair.

Version3 repairs a false-positive likelihood check, not the policy. Native
probe `t135853` replayed the first full round: scalar-versus-saved logprob error0,
maximum batch-versus-scalar error0.0000314713 (mean difference0.00000190735,
std0.185932). The float32 replay tolerance is0.0002, a probability-ratio error
about0.02%, far below PPO's20% clipping range. Performance thresholds are unchanged.
Version2 full is excluded; version3 preflight uses the full16paths/update/method.

## Method

- Reuse the shared SMDP-PPO trainer. Joint updates both actors and both reward
  critics; flat and causal forecast update their lower actor/critic only.
- Keep the full Stage112 strong lower function, its standard deviation, the
  forecaster and native task frozen. Lower input is all392 feedback coordinates,
  four forecast-advice coordinates, four curve-minus-forecast position/velocity
  coordinates, and option-age/remaining-time clocks. Upper receives390 causal
  coordinates plus the same clocks and outputs eight Bernstein coordinates.
- Upper mean and new lower readout start at zero. Execution is frozen teacher
  mean plus donor reference response bounded at0.05 plus a learned residual
  bounded at0.05. Both corrections are inside the Gaussian likelihood used by PPO.
  Zero mean reproduces source forecast exactly; sampled upper exploration is
  new and fixed at standard deviation0.15. Lower noise is unchanged.
- Gamma/lambda1 use full-episode MC credit, actual upper durations50/100, and
  episode-only lower boundaries. Critics start at remaining steps, a known
  upper bound for the fixed-horizon `exp(-distance)` return. No evaluation fit.
- PPO: learning rate0.0003, two epochs, minibatch1024, clip0.2, entropy0,
  max gradient norm1. Optimizer shuffles use recorded root/period/update seeds.

## Experiment

All8 inherited roots, both periods, fresh Stage121 scenarios. Each method gets
8updates x8scenarios x2noise rollouts x1200steps per root/period. Final update8
only; no iteration selection. Evaluation has32 paired fresh scenarios and
seven variants: source flat/forecast, trained flat/forecast/joint, joint with
upper disabled, joint upper transferred to trained forecast lower.

Total:9,728episodes,11,673,600native steps. Environment budgets match, not parameter
counts or total compute. Native donor calls and optimizer steps are recorded.
The two additional frozen donor queries are evaluated even when the delta is zero.

Ten endpoints use equal-root bootstrap65,536, Bonferroni correction. Both
periods must have joint-minus-trained-flat/forecast lower CI above0.5episode
return, and joint-minus-own-blinded/source-forecast lower CI above0. Upper
transfer is a registered mechanism endpoint, not a substitute for these gates.
No automatic seed extension. Preflight is mechanical only; settings are frozen
before it and are not tuned using its returns.

## Execution And Limits

Use scheduler dynamic placement on node001-node006: four workers+parent/4GiB
preflight; four workers+parent/8GiB full. Raw training arrays stay in server RAM.
Only final inference weights are saved server-side; pull compact JSON and logs.
No changes to original FreqDuet/TransitDuet. This tests learned joint plan/control,
not learned promotion, strict frequency responsibility or domain-general superiority.
Stage67 HOLD and the Stage120 result stay unchanged.

## Run Receipt

Code revision `a7f220eb70`:53tests and3subtests passed. Version2 native preflight
`t135835/t135836` passed104episodes/124,800steps at horizon1200.
Version2 full run `pointmaze_joint_reference_stage121_full_20261006_r2` was
stopped for the overly strict likelihood check. Both excluded full runs retain
abort records. Version3 admission requires its full-batch native preflight.
Full performance conclusions await all8 final evaluations and corrected CIs.
