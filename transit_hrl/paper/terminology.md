# Freq-HRL terminology ledger

Status: authoritative manuscript terminology, 2026-08-30

This ledger fixes the vocabulary used by `manuscript.md`. Terms from retired
drafts are not interchangeable with the definitions below.

| Canonical term | Definition | Avoid |
|---|---|---|
| Freq-HRL | The two-timescale actor-critic protocol, frequency-responsibility diagnostics, and guarded restoration procedure implemented in this repository. | Calling every frequency encoder or heuristic router Freq-HRL. |
| upper policy | The policy acting once per macro interval. | Slow policy, planner, and manager as unqualified synonyms. |
| lower policy | The policy acting at each environment step on residual control. | Fast policy and worker as unqualified synonyms. |
| frequency responsibility | The assignment of low-frequency control effects to the upper level and high-frequency residual effects to the lower level. | Physical frequency separation; these are different claims. |
| responsibility-space lower-LF drift | Low-frequency power computed after the registered causal responsibility operator. | Raw lower-LF drift. |
| raw lower-LF drift | Low-frequency power of the pre-routing lower action/effect. | Responsibility-space drift. |
| upper-HF action power | High-frequency power of the upper action/effect under the registered causal filter. | Upper instability or noise without a metric definition. |
| function-preserving router transaction | A change in responsibility coordinates that exactly preserves paired executed-action, reward, and latent-policy traces. | Learned policy improvement or behavioral improvement. |
| guarded actor transaction | A finite actor update admitted only after the registered design-fold and reward-floor gates pass. | Guaranteed-safe update; the guarantee is empirical on the frozen path panel. |
| guarded restoration portfolio | The fixed registry of router and actor transactions plus the common selection and abstention contract. | Ensemble, mixture policy, or end-to-end learned gate. |
| frequency-violation merit | The registered non-negative aggregate excess above the frequency budgets. | Leakage loss unless referring to the training loss itself. |
| worst frequency violation | The largest registered normalized frequency-budget excess in a snapshot. | Worst-case guarantee outside the evaluated paths. |
| reward floor | The paired held-out return condition encoded by zero `reward_violation_count`. | Reward improvement. |
| exact trace invariance | Equality of paired executed-action, reward, and latent-policy trace identifiers, with zero observed return and mean-reward delta. | Approximate behavioral equivalence. |
| design paths | Frozen paths used for candidate eligibility and ranking. | Training paths or validation paths. |
| validation paths | Disjoint frozen paths used once to adjudicate a selected transaction. | Independent optimizer replicates; paths are nested within a seed cell. |
| optimizer-seed replicate | One independently trained anchor policy for a registered optimizer seed. | Treating paths from one anchor as independent replicates. |
| abstention | No transaction passed the design gate; counted as a confirmatory failure. | Missing data or a skipped cell. |
| supported | The exact registered decision rule passed. | Proven, universally valid, or state of the art. |
| noninferiority | The lower confidence bound exceeded the registered negative margin. | Equality or superiority. |
| confirmatory | Source-bound protocol and decision rule frozen before access to the corresponding outcome. | Any large run or post-hoc analysis. |
| development | Evidence used to design or reject mechanisms, excluded from headline confirmation. | Validation or confirmation. |
| Quant v7.4 | The registered synthetic time-series control comparison with 24 independent training replicates. | Real-market deployment evidence. |

## Claim hierarchy

1. **Representation claim:** the implementation causally exposes slow and fast
   state features. This is an implementation property, not a control result.
2. **Responsibility claim:** the registered responsibility-space diagnostic is
   reduced while the reward floor holds. MuJoCo v12 and v14.29 support bounded
   forms of this claim.
3. **Raw behavioral claim:** raw lower-LF drift and upper-HF power both satisfy
   their gates. MuJoCo v13 does not support this claim across all three tasks.
4. **Performance claim:** Freq-HRL improves return over matched learned
   baselines. The current evidence is mixed and includes one supported harm.
5. **Domain-general claim:** the same learned algorithm improves multiple real
   domains. The authoritative ledger does not support this claim.
