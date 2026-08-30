# Freq-HRL submission-readiness decision

Date: 2026-08-30

Decision: **not ready for a top-tier CS conference or journal in its current
scientific form**. The implementation and evidence governance are substantial,
but the main algorithmic claim is still weaker than the likely novelty and
impact bar.

## What is now solid

- The shared upper/lower actor-critic implementation is real, not a heuristic
  protocol.
- MuJoCo v12 provides fresh-seed confirmatory support for responsibility-space
  drift reduction with return noninferiority in three tasks.
- MuJoCo v13 is a valid negative confirmation that prevents raw-behavior
  overclaiming.
- MuJoCo v14.29 is preregistered, source-bound, complete, and statistically
  supported in all three tasks under its exact restoration claim.
- Quant v7.4 uses independent training replicates, guards against path-level
  pseudoreplication, and reports all 12 multiplicity-controlled contrasts.
- The authoritative registry separates confirmatory, development, legacy, and
  excluded evidence and now supplies one manuscript source of truth.

## Blocking scientific issue

The strongest new v14.29 result is mainly a coordinate-restoration result. All
32 HalfCheetah/Hopper cells and six Walker cells selected a router that exactly
preserved executed actions, rewards, and latent policies. This is excellent
mechanism isolation, but it means the result can be interpreted as correcting
responsibility bookkeeping rather than learning better hierarchical control.
The interpretation is strengthened by v13, where the universal raw-action claim
failed, and by Quant v7.4, where generic HRL-GRU-PPO has significantly better
return.

No amount of additional v14.29 seeds resolves this issue. The next experiment
must change the estimand from responsibility coordinates to raw upper/lower
behavior.

## Required algorithmic advance

Implement **behavior-preserving responsibility distillation** as the next frozen
development line:

1. On anchor trajectories, causally decompose the executed total action into a
   smooth upper plan curve and a high-pass lower residual.
2. Distill those targets into the raw upper and lower actor outputs, not only the
   responsibility reporter.
3. Penalize total-action reconstruction error and upper boundary jumps during
   distillation; retain the paired reward floor in closed-loop selection.
4. Gate candidates on raw lower-LF reduction, upper-HF power, reconstruction,
   and reward simultaneously on every design fold.
5. Validate once on disjoint paths and count abstention as failure.
6. Expand to fresh-seed confirmation only if one fixed algorithm passes all
   three environments in development. A per-environment post-hoc mechanism is
   insufficient for the domain-general claim.

This directly targets the v13 failure modes: HalfCheetah's lower actor must lose
its slow bias, while Hopper's upper transfer must become smooth enough to remain
inside the high-frequency budget.

## Evidence still needed after the algorithmic advance

- A fresh-seed MuJoCo confirmation of joint raw lower-LF reduction, upper-HF
  control, reconstruction, and return noninferiority in all three tasks.
- A Quant rerun against generic HRL-GRU-PPO showing at least noninferiority on
  return; otherwise the paper must remain a diagnostics paper.
- One authoritative non-synthetic second-domain result. Current Transit and
  order-book artifacts cannot be used until registered with valid inferential
  units and outcome-level control endpoints.
- An ablation that separates the causal encoder, two-rate policy, raw
  distillation, reward guard, and abstention rule.
- A formal proposition stating when exact total-action reconstruction preserves
  return pathwise and why that does not imply raw policy equivalence; a global
  RL convergence theorem is not required.
- Replacement figures generated only from the authoritative ledger. The current
  `manuscript_figures_latest` package predates v14.29 and includes evidence that
  is not reportable under the current registry.

## Current paper positioning

The strongest honest title is **"Freq-HRL: Auditing and Guarded Restoration of
Frequency Responsibility in Hierarchical Reinforcement Learning."** The paper
can currently be developed as a rigorous diagnostics/restoration paper. It
should not use "domain-general," "performance improvement," "no tradeoff," or
"physical frequency separation" as an unqualified headline claim.

The replacement manuscript is `transit_hrl/paper/manuscript.md`. The retired
2026-06-27 draft remains only as provenance and must not be submitted.
