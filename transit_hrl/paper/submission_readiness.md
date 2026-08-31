# Freq-HRL submission-readiness decision

Date: 2026-08-31

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
  excluded evidence and now supplies one manuscript source of truth. Its current
  47 records comprise 4 reportable confirmatory records, 41 development-only
  records, and 2 excluded legacy records; only 2 support positive claims.
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
has significantly better return. The v17.8--v18.5 sequence also failed to turn
the responsibility result into a trusted causal total-action correction.

No amount of additional v14.29 seeds, expansion of the same output-head grid,
or further tuning on the reused 120-path panel resolves this issue. The next
experiment must change the training architecture and use a genuinely fresh
panel, not only alter the post-training coordinate.

## Closed development line

The v16--v18.5 sequence now bounds the direct extensions of the restoration
result:

- v17.4 showed that a streaming responsibility projection can reduce lower-LF
  and joint merit while preserving behavior, but the absolute component budgets
  were not jointly met in all environments.
- v17.6 established that 81 of 88 online failures were recoverable by an acausal
  fixed-total-action oracle. Seven Hopper paths required total-action changes.
- v17.11 recovered only 62 of 81 router-recoverable paths and only 14 of 33 in
  Hopper, closing the frozen router-only FIR line.
- v17.14 recovered six of seven actor-floor paths with a frozen linear causal
  FIR while preserving all 113 reference-feasible paths. One Hopper `ood_chirp`
  path remained unresolved, so the linear actor-residual grid was closed.
- v18.2 state conditioning recovered only three of seven actor-floor paths.
- v18.3 recovered all seven by instantaneous projection, but only with action
  corrections far outside the frozen trust region.
- v18.4 exposed an offline-exact versus online-direct failure: 120 corrected
  totals were oracle-feasible, but only 69 realized component traces were
  directly feasible, with 40,962 prefix budget violations.
- v18.5 found no preregistered target-free floor score eligible to gate another
  feedback screen.

No v17.8--v18.5 experiment accessed fresh validation paths. A post-hoc
combination of v18.5 scores cannot authorize another tuned screen. The unchanged
120-path panel is now retired for selection; it may be used only for documented
failure analysis.

## Required algorithmic advance

The remaining top-tier route is a newly frozen **training-time physical
Freq-HRL** architecture:

1. Change the total-action generator directly rather than rely on a
   function-preserving responsibility router.
2. Enforce joint upper/lower frequency feasibility with a recursive terminal
   certificate or invariant set, so a receding horizon cannot move unresolved
   frequency debt beyond its boundary.
3. Keep the correction target-free at execution time and inside a frozen action
   trust region; do not select architecture or thresholds per environment.
4. Freeze the architecture before training on optimizer seeds and path roots
   that are disjoint from every v15--v18.5 development artifact.
5. Gate one fixed candidate on raw lower-LF drift, upper-HF power,
   responsibility drift, exact total reconstruction, return noninferiority, and
   correction magnitude on every design fold.
6. Enter fresh-seed confirmation only after the same fixed architecture passes
   all three MuJoCo environments. Development success alone remains ineligible
   for the manuscript headline.

## Evidence still needed after the algorithmic advance

- A fresh-seed MuJoCo confirmation of joint raw lower-LF reduction, upper-HF
  control, reconstruction, and return noninferiority in all three tasks.
- A Quant rerun against generic HRL-GRU-PPO showing at least noninferiority on
  return; otherwise the paper must remain a diagnostics paper.
- One authoritative non-synthetic second-domain result. Current Transit and
  order-book artifacts cannot be used until registered with valid inferential
  units and outcome-level control endpoints.
- An ablation that separates the causal encoder, two-rate policy, gauge layer,
  physical correction, reward guard, and abstention rule in any newly frozen
  training-time architecture.
- A recursive-feasibility argument for the online frequency constraint. The
  formal gauge propositions and empirical pathwise invariance checks are already
  sufficient for the bounded responsibility claim, but they do not certify a
  physical receding-horizon controller.
- Replacement figures generated only from the authoritative ledger. The current
  `manuscript_figures_latest` package predates v14.29 and includes evidence that
  is not reportable under the current registry.

## Current paper positioning

The strongest honest title is **"Freq-HRL: Auditing and Guarded Restoration of
Frequency Responsibility in Hierarchical Reinforcement Learning."** The paper
is now coherent as a rigorous diagnostics/restoration submission once its
figures, citations, and supplementary evidence package are rebuilt from the
authoritative registry. It is still below the top-tier algorithmic bar without
the physical training-time advance above. It should not use "domain-general,"
"performance improvement," "no tradeoff," or "physical frequency separation"
as an unqualified headline claim.

The replacement manuscript is `transit_hrl/paper/manuscript.md`. The retired
2026-06-27 draft remains only as provenance and must not be submitted.
