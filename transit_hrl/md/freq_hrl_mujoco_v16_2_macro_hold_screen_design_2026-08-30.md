# MuJoCo v16.2 Macro-Hold Gauge Development Screen

## Question

Does enforcing the upper responsibility at the actual upper decision rate
repair the primitive adaptive gauge's upper-HF failure while preserving reward
and reducing lower-LF drift?

## Frozen Matrix

- Three environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5.
- Three fresh optimizer seeds per environment.
- Three capacity-matched reward-only arms: direct, primitive adaptive gauge,
  and macro-hold adaptive gauge.
- Four fresh training roots, four fresh checkpoint-selection roots, and eight
  fresh held-out roots crossed with five disturbance families.
- 27 independent scheduleurm cells on node001-node006 with dynamic placement.

All arms use the same actor-critic architecture, hidden width, training budget,
optimizer, state inputs, cost-critic capacity, and reward-only checkpoint
selector. Only the responsibility router changes.

## Frozen Gate

For each environment-by-optimizer-seed unit, the macro-hold candidate must:

1. retain at least 95% of the direct arm's held-out reward;
2. reconstruct the additive responsibility within RMS `1e-7`;
3. incur zero router clipping, so the upper responsibility is genuinely held;
4. keep mean upper HPF8 power below `0.075^2`;
5. reduce lower LPF32 power by at least 10% versus its own latent split; and
6. reduce normalized joint upper-HPF/lower-LPF merit by at least 10% versus its
   own latent split.

An environment passes with at least two of three optimizer seeds. Expansion is
authorized only if all three environments pass. Primitive-gauge component
effects are retained as diagnostics and cannot replace a failed primary gate.

## Evidence Boundary

This is frozen development evidence, not confirmation. A supported result
authorizes a fresh-seed confirmatory protocol; it does not itself enter the
paper as a confirmatory performance claim.
