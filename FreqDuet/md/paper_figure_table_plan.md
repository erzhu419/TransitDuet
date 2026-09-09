# FreqDuet Paper Figure And Table Plan

Last updated: 2026-09-10 CST

## Status

The June curation bundle is retained as a historical package. The current
paper-facing source is the Protocol V6 current-best evidence bundle. It must
show the successful V8 confirmation and failed V9 long-training gate together.

## Required V6 Main Tables

| Table | Required contents | Claim gate |
| --- | --- | --- |
| 1. V8 confirmed effect | confirmed main versus no guard; journey, wait, in-vehicle time, headway CV, holding, denial, unserved, and service cost | Six training x four untouched evaluation seeds |
| 2. V9 long-training robustness | Same metrics and controller at 200 episodes, with gate status shown | Eight training x eight evaluation seeds; negative result visible |
| 3. V9 external trade-off | confirmed main versus fixed headway, rule holding, and rule MPC | Same source/scenario contract and paired intervals |
| 4. Evidence decisions | Experiment stage, seed counts, gate decision, and claim eligibility | Prevents V8/V9 or historical V1 pooling |

The old composite can appear only in an appendix sensitivity table.

## Required V6 Main Figures

| Figure | Panels | Evidence |
| --- | --- | --- |
| 1. Method | causal decomposer, LF upper state, HF lower state, exact headway planner, holding guard, passenger-journey credit | Source-bound architecture diagram |
| 2. Main result | V8 paired journey and headway-CV deltas versus no guard | V8 independent confirmation only |
| 3. Robustness and baselines | V9 long-training result beside fixed/rule/MPC trade-offs | V9, visibly marked as failed long-training gate |
| 4. Causal/physical audit | plan budget, effective launch shifts, holding passenger-min, denied-trip/readiness outcomes | Protocol V6 traces only |
| 5. Generalization/realism | held-out perturbations plus clearly separated AFC/APC/AVL realism panels | No field-effect overclaim |

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
