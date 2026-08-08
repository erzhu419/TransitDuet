# Freq-HRL MuJoCo v13 Behavioral Confirmatory Protocol

Date: 2026-08-08

## Purpose

MuJoCo v12 supported return noninferiority and a reduction in
`LowerLFDriftAbs`, but that leakage endpoint measures controller
responsibility rather than physical policy behavior. The v13 experiment adds
two external behavioral gates while retaining the v12 responsibility and
return gates.

The endpoint set and numerical thresholds were developed after exploratory
inspection of v12. This is disclosed prospectively. Every v13 optimizer,
training, checkpoint-selection, safety-selection, and held-out evaluation seed
is fresh and disjoint from v12. No v13 held-out result may be accessed before
this protocol, runtime, launcher, and analyzer are committed and frozen.

## Frozen Comparison

- environments: `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5`;
- baseline: `freq_hrl_no_leakage` with additive responsibility;
- full method: `freq_hrl_safe_selector` with causal low-frequency transfer;
- 24 paired optimizer replicates per environment and arm;
- 4 training seeds, 4 checkpoint-selection seeds, and 12 safety-selection
  seeds per optimizer replicate;
- 8 held-out seeds under standard, low-frequency, high-frequency, mixed, and
  OOD-chirp disturbances;
- 72 independent cells per arm and 144 cells in total;
- frozen algorithm revision:
  `8e47614f1005d8a064a3d6691a0ca6e5bb311ee4`;
- frozen source manifest:
  `002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7`.

The paired baseline checkpoint inside each full-method safe selector must be
byte-identical to the corresponding externally run baseline checkpoint.

## Primary Endpoints

Four one-sided gates are evaluated separately in each environment:

1. held-out episode return is noninferior within 2% of the absolute baseline
   mean return;
2. mean `LowerLFDriftAbs` is reduced by at least 10% relative to baseline;
3. mean `RawLowerLFDriftAbs` is reduced by at least 10% relative to baseline;
4. full-method upper-controller high-frequency RMS is at most 0.10, where the
   endpoint is `sqrt(mean(UpperHFPowerAbs))`.

Paired optimizer replicates are the independent resampling unit. A fixed
50,000-draw paired bootstrap is used. The family-wise alpha is 0.05 across all
12 statistical gates, giving one-sided confidence
`1 - 0.05 / 12 = 0.995833...` per gate. The decision is supported only if all
12 gates pass.

`ResponsibilityReconstructionRMS <= 1e-7` in every environment is an additional
deterministic integrity requirement, not one of the 12 statistical gates.
Return superiority and ordinary 95% intervals are exploratory diagnostics and
cannot replace a failed primary gate.

## Claim Boundary

If all gates pass, the supported statement is:

> On three continuous-control environments, the frozen Freq-HRL full method
> reduced both responsibility-space and raw lower-controller low-frequency
> drift, bounded upper-controller high-frequency responsibility, and met a
> family-wise return-noninferiority criterion against a checkpoint-matched
> additive-responsibility baseline.

This experiment does not establish universal action smoothing, optimal
frequency decomposition, or superiority over every non-frequency HRL method.
Any failed environment or endpoint remains visible and narrows the claim.

## Execution And Access Control

The two arms use separate run directories:

- `mujoco_v13_behavioral_confirmatory_baseline_20260808_r1`;
- `mujoco_v13_behavioral_confirmatory_full_20260808_r1`.

Each one-core cell is submitted through scheduleurm with dynamic placement over
`node001` through `node006`; no task is pinned to one node. `jtl110cpu` and
`jtl110cpu2` are excluded. Partial outcome files are counted only for
completion monitoring and must not be analyzed or used to change the protocol.

After all 144 cells have completed and synced, each arm is merged and the
committed analyzer is run once against the common frozen runtime revision. The
decision JSON, paired rows, environment gate table, source identities, and
artifact hashes are retained for the manuscript evidence registry.
