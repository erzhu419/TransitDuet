# Freq-HRL v7.4 HPO Selection Repair

Date: 2026-08-08

## Preserved Failure

The source-bound 210-cell final HPO matrix completed under algorithm revision
`5f54bb2323e5cbeb1a5beea6548324c01c131085` and source manifest
`56c663db9a391a05dbab3e097c305de60b7e251a36451633dd4934853644aa30`.
The original merge is preserved unchanged with status
`provisional_support_validation_only`.

All seven independently tuned variants passed the learning gate, and the full
Freq-HRL candidate passed the mechanism-activity gate. Five selected variants
passed the registered training-budget gate. The original selector nevertheless
chose budget-ineligible rank-1 candidates for both generic-HRL baselines:

| Variant | Original candidate | Plateau fraction | Budget status |
| --- | --- | ---: | --- |
| generic HRL PPO | `ppo_lr3e4_std15` | 0.60 | unstable tail |
| generic HRL GRU-PPO | `ppo_lr3e4_std10` | 0.40 | unstable tail |

The merge then correctly refused to freeze because the selected models failed
the budget gate.

## Implementation Defect

The final validator requires every selected candidate to satisfy:

1. learning eligibility;
2. mechanism eligibility where applicable;
3. sufficient training-budget evidence.

The candidate-selection pool filtered only the first two conditions. A
budget-ineligible high-LCB candidate could therefore block an already evaluated
and fully eligible lower-ranked candidate. This is an analysis-selection defect,
not evidence that the valid candidate failed to learn.

## Source-Preserving Repair

`scripts/repair_full_method_hpo_v74_selection.py` re-merges all 210 original
cells and applies all three existing eligibility conditions before retaining
the original LCB ordering. It does not change any score, threshold, seed,
training artifact, or algorithm byte, and it does not load OOD, promotion-
recovery, or held-out confirmatory paths.

The expected replacements are:

| Variant | Repaired candidate | Original rank | Plateau fraction |
| --- | --- | ---: | ---: |
| generic HRL PPO | `ppo_lr3e4_std10` | 3 | 0.80 |
| generic HRL GRU-PPO | `ppo_lr3e4_std05` | 3 | 0.80 |

All other selected candidates must remain unchanged. The five mechanism
ablations continue to inherit the full-method candidate.

The repaired output is written to a new directory; the original failed merge
is never overwritten. The output records a manifest SHA-256 over all merge
inputs, the committed repair runtime SHA/revision, before/after selections, and
held-out access status. It is usable only if the unchanged v7.4
`validate_frozen_config` independently returns `valid`.
