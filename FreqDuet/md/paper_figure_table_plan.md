# FreqDuet Paper Figure And Table Plan

Last updated: 2026-09-10 CST

## Status

The June curation bundle is retained as a historical package. The current
paper-facing source is the Protocol V6 current-best evidence bundle. It must
show the successful V8 confirmation and failed V9 long-training gate together.

## Required V6 Main Tables

| Table | Required contents | Claim gate |
| --- | --- | --- |
| 1. V8 confirmed effect | confirmed main versus the Protocol V6 reference config; journey, wait, in-vehicle time, headway CV, holding, denial, unserved, and service cost | Six training x four untouched evaluation seeds; both configs have the legacy holding guard disabled |
| 2. V9 long-training robustness | Same metrics and controller at 200 episodes, with gate status shown | Eight training x eight evaluation seeds; negative result visible |
| 3. V9 external trade-off | confirmed main versus fixed headway, rule holding, and rule MPC | Same source/scenario contract and paired intervals |
| 4. Evidence decisions | Experiment stage, seed counts, gate decision, and claim eligibility | Prevents V8/V9 or historical V1 pooling |

The old composite can appear only in an appendix sensitivity table.

## Required V6 Main Figures

| Figure | Panels | Evidence |
| --- | --- | --- |
| 1. Method | historical harmonic prior, causal LF/HF split, LF upper state, HF plus compact APC/AVL lower state, executable headway planner, discrete holding, and frozen pre-action two-sided regularity reward | `fig1_protocol_v6_method`; source-bound to the packaged resolved config; legacy holding guard, promotion, and leakage are shown as disabled |
| 2. Main result | V8 and V9 paired journey/headway-CV deltas versus the Protocol V6 reference | `fig2_protocol_v6_confirmation_robustness`; stages remain separate, V9 is marked not confirmed, and the contrast is not called a legacy-guard effect |
| 3. Robustness and baselines | Fixed/rule/MPC journey, headway-CV, and service-cost trade-offs | `fig3_protocol_v6_external_tradeoff`; source-identical V9 comparison only |
| 4. Causal/physical audit | vehicle holding, passenger holding, denied-trip rate, and terminal execution error | `fig4_protocol_v6_physical_outcomes`; paired V8/V9 full-policy effects, not an isolated-module claim |
| 5. Generalization/realism | separately normalized FreqDuet OD plus balanced complete subsets of the bounded public MTA AFC and Halifax APC caches | `fig5_protocol_v6_external_realism`; descriptive unmatched-source realism audit only, not a population estimate |

## Appendix Requirements

- V4 corrected negative result and old objective mismatch.
- Full single-axis config diff and source/seed manifests.
- All unsuccessful V6 candidates and stopping decisions, including V9 and V28-V32.
- Per-seed rows, paired bootstrap details, sign-flip tests, Holm corrections,
  and effect sizes.
- External-data provenance, license/access boundaries, and exact distinction
  between realism audit and field validation.

## Promotion Rule

`build_freqduet_protocol_v6_evidence_package.py` builds draft-facing tables and
source artifacts without overriding the submission hold. The legacy package
builder remains historical. A final submission package still requires an
explicit decision about the failed V9 long-training gate; a zero-missing-file
report cannot change that scientific result.
