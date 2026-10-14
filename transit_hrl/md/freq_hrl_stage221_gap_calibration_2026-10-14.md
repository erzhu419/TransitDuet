# Stage221: cost-gap objective and train-future wait shrinkage

Stage220's hard safe-winner CE lost cost magnitudes. Its wait residuals lowered training risk but increased risk even on other futures of the same prefixes. This stage tests both causes without enlarging the cost representation.

The original 240 states, four matched single/mixed views, 60/60-state complete-prefix folds, two future partitions and both directions are unchanged. Ridge, CE and raw wait heads remain controls. A new 98-parameter ridge-initialized residual minimizes KL to a soft target based on physical mean cost differences, without a true-safe action mask. Temperature is the outer training half's nonidentity cost RMS.

Each wait architecture gets two complementary 2/2-future inner fits within the outer training four draws. A scalar alpha in [0,1] minimizes pooled inner out-of-fit physical draw-mass log-loss. The final four-draw residual is multiplied by that alpha. No outer held prefix or evaluation-half label selects alpha, temperature or optimizer steps.

Budget: eight fit tasks, 32 cells, 32 analytic cost fits, 96 GLM fits, 38,400 CE/gap updates, 172,800 wait residual updates and 96 scalar calibrations. Both cost objectives have 600 updates; all three wait architectures have the same full and inner update budgets. CPU1/RAM3072 tasks are unbound across node001-node006. Models and predictions stay remote; only the compact final report is retrieved.

## Status

Implemented in the isolated Freq-HRL checkout. Six new focused tests and six Stage220 control tests passed (12/12): cost-gap magnitude/ties, convex shrinkage, causal inference, future/held-label isolation, full report coverage and scheduler budgets. Scheduler fitting and results pending.

## Next Decision

Compare same-gate ridge/CE/gap selected physical cost and wait, and raw/shrunk proper wait risk in train_same, train_other and held_other. Retain conditioning if its selected-action benefit survives shrinkage, rather than treating lower global log-loss as sufficient control success.

## Limitations

Inner calibration holds out futures, not prefixes; a two-draw shrinkage factor may transfer imperfectly to the final four-draw fit. Both partitions reuse eight conditional draws. This is one-action supervised validation with the frozen greedy suffix, not independent-seed CI or end-to-end learned HRL.
