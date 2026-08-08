# Freq-HRL MuJoCo v4 Pilot Audit And v5 Rebuild

Date: 2026-08-08

## Decision

The source-bound v4 pilot is integrity-valid but performance-negative. It must
not be scaled into a confirmatory experiment and must not be cited as evidence
that Freq-HRL is generally superior. Protocol v5 is a structural repair, not a
post-hoc relabeling of the v4 result.

## v4 Integrity

- 36/36 cells completed on `node001` through `node006`.
- The matrix contains three environments, four methods, and three independent
  optimizer replicates per method/environment.
- The audit found 0 integrity issues and 0 warnings.
- Every cell has a uniform verified source identity, complete evaluation rows,
  an independent parameter hash, and a verified serialized-checkpoint hash.
- The evidence role is `development_only_not_claim_eligible`.

## v4 Standard-Task Result

| environment | flat PPO | generic HRL | Freq-HRL no leakage | Freq-HRL |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 831.27 | 110.37 | 151.03 | 77.39 |
| Hopper-v5 | 195.45 | 150.12 | 135.67 | 123.33 |
| Walker2d-v5 | 249.42 | 271.58 | 265.10 | 269.47 |

The values are means over three independent optimizer replicates after each
replicate averages its five held-out episode paths. Freq-HRL loses to flat PPO
on HalfCheetah and Hopper and does not consistently beat generic HRL. Leakage
also hurts HalfCheetah. Increasing only the number of seeds or updates would
not repair this failed development gate.

## Structural Causes

1. v4 decomposed the complete MuJoCo observation. That observation is mainly
   endogenous physical state, not an exogenous time series. Frequency routing
   therefore removed absolute posture and velocity information from the lower
   controller, contrary to the Freq-HRL interface contract.
2. The lower residual had scale 0.35 while the upper anchor was held for 16
   primitive steps. This denied the fast controller enough authority to learn
   a locomotion gait.
3. Training used only the undisturbed task. The frequency channels therefore
   contained no explicit exogenous training signal even though evaluation
   added low-, high-, mixed-, and chirp-frequency disturbances.
4. The leakage multiplier started at 0.05. Because cost advantages were
   normalized, even a numerically tiny positive cost could create a material
   actor gradient from an uncalibrated cost critic.
5. MuJoCo is a secondary generalization domain. Its actuation-disturbance
   benchmark cannot replace the native time-series evidence in Trading and
   Transit.

## v5 Contract

Protocol `freq_hrl_mujoco_shared_core_v5` makes the following changes:

- raw endogenous physical state is provided to every policy level;
- only the explicit, currently observed causal actuation disturbance is
  decomposed into slow, middle, and high bands;
- Freq-HRL routes slow/middle disturbance context upward and middle/high
  context downward, while generic HRL receives raw disturbance and its first
  difference at both levels;
- standard, low-frequency, high-frequency, and mixed disturbances are assigned
  to disjoint registered training and checkpoint-selection seeds; the chirp
  remains an OOD evaluation condition;
- lower control scale is 1.0 and upper anchor scale is 0.35;
- the leakage dual starts at zero and adapts from observed violation;
- the cost critic output is zero-initialized and receives no optimizer update
  while the constraint is inactive;
- mean squared budget excess below `1e-6` cannot activate a cost-policy
  gradient;
- every seed-to-condition assignment and both checkpoint hashes are written to
  the cell artifact.

## Required Gates

Before a v5 pilot may be expanded:

1. all unit and integration tests must pass;
2. under the standard zero-disturbance condition, `freq_hrl_no_leakage` and
   `generic_hrl` must produce identical checkpoints for the same optimizer and
   environment seeds;
3. an inactive leakage constraint must leave the lower actor identical to the
   unconstrained actor;
4. the source-bound preflight must complete all four methods with zero audit
   issues and zero warnings;
5. the 36-cell pilot must improve the frequency-versus-generic comparison in
   the disturbed conditions without destroying standard-task performance.

Even a successful v5 pilot remains development evidence. Formal evidence needs
fresh optimizer seeds, untouched held-out seeds, a frozen source manifest,
approximately one million or more primitive training transitions per
replicate, at least ten independent optimizer replicates, and a registered
multiplicity-controlled analysis.
