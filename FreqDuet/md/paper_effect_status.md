# FreqDuet Paper Effect Status

Last updated: 2026-09-10 CST

## Current Result

The current best physically and causally valid controller is
`F_freqduet_protocol_v6_confirmed_main_hiro`. It is an exact naming alias of the
compact-AVL, two-sided regularity, weight-two policy confirmed in V8. V28-V32
did not pass their development gates and do not alter this controller.

V8 independently confirmed the 40-episode primary result on six training seeds
crossed with four untouched evaluation seeds. Relative to the same-semantics
no-guard controller, headway CV changed by `-0.02231`, 95% CI
`[-0.03805,-0.00750]`. Restricted passenger journey changed by `-0.26266 min`,
95% CI `[-0.83661,+0.17494]`, satisfying the registered no-harm criterion but
not establishing a significant journey reduction.

## Required Negative Result

The independent 200-episode V9 matrix did not confirm the registered long-run
headway effect. Journey versus no guard improved by `-1.24238 min`, 95% CI
`[-2.20444,-0.53635]`, but headway CV changed by only `-0.00911`, 95% CI
`[-0.02785,+0.00572]`. The gate status is `longtrain_not_confirmed`.

The source-identical V9 external comparison is a trade-off result. Relative to
fixed headway, FreqDuet reduced restricted service cost by `-0.12194` and
headway CV by `-0.21928`, but increased restricted passenger journey by
`+2.47025 min`, 95% CI `[+1.87431,+3.18799]`. It reduced journey time relative
to rule holding by `-3.22404 min` and rule MPC by `-26.51111 min`.

## Submission Status

**HOLD.** The evidence is suitable for drafting a transparent result section,
but the registered 200-episode gate failed. The manuscript may report the V8
confirmed effect together with the V9 robustness failure; it may not describe
the controller as long-run confirmed or as outperforming fixed headway on
passenger journey time.

The June V1/composite package remains historical. Its known-domain action
selection and pre-audit physical protocol cannot be used as current headline
evidence.
