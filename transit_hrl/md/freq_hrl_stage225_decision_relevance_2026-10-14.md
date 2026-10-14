# Stage225: wait-loss decision relevance

Stage224 fixed final training floors but worsened held outcomes. Diagnose the objective before fitting another model.

## Protocol

Reuse exact Stage224 final4 models/calibration and Stage223 c2v4 OOF predictions. Shared causal-cost subsets: all five candidates, negative-cost contenders, the unfiltered negative-cost front runner, and excluded nonnegative-cost candidates. Original-action ties stay original. Compare proper loss, false-safe harm, false-unsafe missed wait benefit, and residual-shrinkage gradient contributions.

Use both future halves to separate stable wait-sign errors from sign flips and stable switch harm from half-specific harm. The true-wait-menu selector keeps predicted-cost ranking fixed and is diagnostic only. Aggregate sums/counts, including empty subsets, rather than averaging unequal subset means.

32 scheduler analysis cells plus one report; 1 CPU/3 GB each, unbound node001-node006. Zero training, calibration or rollout. Six scopes: final train/held x same/other futures, plus OOF same/other. Models/panels remain remote; pull only the aggregate report.

## Status

Six focused tests passed: causal masks/ties/empty subsets, loss-gradient reversal, future-switch attribution, count-weighted aggregation, complete coverage/controls, and scheduler/JSON protocol. Formal results pending; default unchanged.

## Next Decision

If irrelevant candidates dominate loss/gradients while contenders disagree, test a causal action-weighted wait objective with matched 600-step/final4 training. If errors remain on contenders, target generalization rather than claim loss dilution solves it. Future sign flips alone do not establish either mechanism.

## Limitations

Repeated development states/eight draws are not independent confirmation seeds. Causal cost subsets depend on a fitted cost predictor; postselection/oracle comparisons are descriptive. Source roots bundle checkpoint and demand/prefix differences. This is not end-to-end learned HRL.
