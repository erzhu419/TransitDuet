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
- Additive raw-policy gauge non-identifiability and causal canonical
  responsibility fixing are now explicit in both code and the method.

## Blocking scientific issue

The strongest new v14.29 result is mainly a coordinate-restoration result. All
32 HalfCheetah/Hopper cells and six Walker cells selected a router that exactly
preserved executed actions, rewards, and latent policies. This is excellent
mechanism isolation, but it means the result can be interpreted as correcting
responsibility bookkeeping rather than learning better hierarchical control.
The interpretation is strengthened by v13, where the universal raw-action claim
failed, by the v15--v15.2 development sequence, where post-hoc raw-policy
distillation passed only Hopper, and by Quant v7.4, where generic HRL-GRU-PPO
has significantly better return.

No amount of additional v14.29 seeds or expansion of the same output-head grid
resolves this issue. The next experiment must change the training architecture,
not only the post-training coordinate.

## Required algorithmic advance

Implement **training-time gauge-fixed Freq-HRL** as the next frozen development
line:

1. Place the shared causal gauge layer inside rollout and training so every
   auxiliary frequency cost is computed in an identifiable coordinate.
2. Route endogenous state causally by level rather than sending the complete raw
   observation to both actors unchanged; retain only the Markov context required
   for stable control.
3. Use separate upper and lower optimization/trust controls. The v15 shared head
   radius coupled HalfCheetah compensation to Walker upper smoothing.
4. Train from fresh optimizer seeds that exclude `2978317753`; v15 trajectory
   roots cannot be recycled as a new confirmation panel.
5. Gate candidates on raw lower-LF drift, raw and responsibility upper-HF power,
   exact total reconstruction, and return on every design fold.
6. Expand to fresh-seed confirmation only if one fixed architecture passes all
   three environments in development. A per-environment post-hoc mechanism is
   insufficient for the domain-general claim.

This directly targets the v15 failure modes: HalfCheetah requires stable lower
compensation under an upper change, while Walker requires substantially stronger
upper smoothing without reintroducing lower drift.

## Evidence still needed after the algorithmic advance

- A fresh-seed MuJoCo confirmation of joint raw lower-LF reduction, upper-HF
  control, reconstruction, and return noninferiority in all three tasks.
- A Quant rerun against generic HRL-GRU-PPO showing at least noninferiority on
  return; otherwise the paper must remain a diagnostics paper.
- One authoritative non-synthetic second-domain result. Current Transit and
  order-book artifacts cannot be used until registered with valid inferential
  units and outcome-level control endpoints.
- An ablation that separates the causal encoder, two-rate policy, gauge layer,
  reward guard, and abstention rule.
- Empirical tests of gauge invariance and pathwise reconstruction across real
  policy traces. The formal propositions are now in the manuscript; a global RL
  convergence theorem is neither established nor required for the bounded claim.
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
