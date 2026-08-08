# FreqDuet Paper Figure And Table Plan

Last updated: 2026-08-08 CST

## Status

The June curation bundle is retained as a historical package. It must not be
used as the current main paper because it is keyed to the legacy composite
objective. Final panel selection starts only after the V5 claim gate passes.

## Required V5 Main Tables

| Table | Required contents | Claim gate |
| --- | --- | --- |
| 1. Main effect | V5 main versus fixed-headway, rule-holding, rule-MPC; restricted journey, wait, in-vehicle time, and normalized safety metrics | Same tapes, crossed CI, untouched confirmation |
| 2. Frequency allocation | main, nofreq, rawhistory, allfreq, upperonly, loweronly | Locked V5 development and confirmation rules |
| 3. Physical mechanisms | nobudget, noguard, noloadcost, waitonlycredit, CSAC | Primary effect plus safety/no-harm diagnostics |
| 4. Generalization | demand noise, OD shift, rush shift, fleet/dwell/service perturbations, route/day splits where available | Held-out seeds and immutable scenario manifests |

The old composite can appear only in an appendix sensitivity table.

## Required V5 Main Figures

| Figure | Panels | Evidence |
| --- | --- | --- |
| 1. Method | causal decomposer, LF upper state, HF lower state, exact headway planner, holding guard, passenger-journey credit | Source-bound architecture diagram |
| 2. Main result | paired journey-time deltas and safety trade-offs versus external baselines | V5 untouched confirmation |
| 3. Frequency mechanism | decomposer recovery, state/action spectra, HF residual to holding lag, LF state to headway-plan response | V5 traces only |
| 4. Causal/physical audit | plan budget, effective launch shifts, guard activation, holding passenger-min, denied-trip/readiness outcomes | V5 mechanism ablations |
| 5. Generalization/realism | held-out perturbations plus clearly separated AFC/APC/AVL realism panels | No field-effect overclaim |

## Appendix Requirements

- V4 corrected negative result and old objective mismatch.
- Full single-axis config diff and source/seed manifests.
- All unsuccessful V5 candidates and stopping decisions.
- Per-seed rows, paired bootstrap details, sign-flip tests, Holm corrections,
  and effect sizes.
- External-data provenance, license/access boundaries, and exact distinction
  between realism audit and field validation.

## Promotion Rule

`build_freqduet_paper_package.py` and `curate_freqduet_paper_panels.py` may be
used for final output only after they are rebound to V5 artifacts and the V5
primary endpoint. A zero-missing-artifact report from the historical manifest
does not satisfy this rule.
