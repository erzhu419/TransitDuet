# Stage225: wait-loss decision relevance

Stage224 fixed final training floors but worsened held outcomes. Diagnose the objective before fitting another model.

## Protocol

Reuse exact Stage224 final4 models/calibration and Stage223 c2v4 OOF predictions. Shared causal-cost subsets: all five candidates, negative-cost contenders, the unfiltered negative-cost front runner, and excluded nonnegative-cost candidates. Original-action ties stay original. Compare proper loss, false-safe harm, false-unsafe missed wait benefit, and residual-shrinkage gradient contributions.

Use both future halves to separate stable wait-sign errors from sign flips and stable switch harm from half-specific harm. The true-wait-menu selector keeps predicted-cost ranking fixed and is diagnostic only. Aggregate sums/counts, including empty subsets, rather than averaging unequal subset means.

32 scheduler analysis cells plus one report; 1 CPU/3 GB each, unbound node001-node006. Zero training, calibration or rollout. Six scopes: final train/held x same/other futures, plus OOF same/other. Models/panels remain remote; pull only the aggregate report.

## Results

Analysis tasks t152107-t152138 and report t152139 completed with exit0. Six tests passed. The 1,280 Stage224 controls reproduce within 1.39e-17. Only the 1,885,658-byte aggregate report was pulled; no raw/models, training, calibration or policy replacement.

Candidate dilution is not a sufficient explanation: all three residual heads have worse held proper loss than GLM on contenders and front runners in all four views. Conditioned contender raw/GLM loss ratios are 1.79, 1.36, 1.40, 1.45 for mixed_a/mixed_b/root397/root401. OOF same-future contender/front gradients at alpha0 are positive in all four views, not opposed by excluded candidates.

Conditioned raw has fewer contender sign mistakes than GLM in all four views, but their mean wait magnitude is not uniformly lower: mixed_a raw0.016233 vs GLM0.009099 min, mixed_b0.016600 vs0.026336, root3970.022476 vs0.021853, root4010.020872 vs0.018009. Neither mistake counts nor overconfidence alone explains the performance gap.

Contender mean-wait signs flip between future halves in 32.6%-33.8% of held cases. The failing joint mixed_a even_odd/1 summary has exactly one switch: wait increment +0.42125 min in evaluated futures, -0.06950 in the other half; cost increments +0.037618/+0.098074 respectively. It is half-specific wait harm but stable cost harm, not exclusively noise.

The diagnostic true-wait menu improves mixed_b wait (-0.015017 vs raw -0.002697 min) but makes cost positive (+0.000628 vs raw -0.000950). Correct observed wait signs with unchanged predicted-cost ranking do not guarantee joint cost/wait improvement.

## Next Decision

Do not launch contender-only wait-loss weighting on this evidence. Next test a joint candidate-to-raw cost/wait advantage objective, with checkpoint selection confined to training-internal prefix/future folds and matched 600-step/final4 budget. Retain raw and c2v4 controls; stop adding training-set alpha floors.

## Limitations

Repeated development states/eight draws are not independent confirmation seeds. Causal cost subsets depend on a fitted cost predictor; postselection/oracle comparisons are descriptive. Source roots bundle checkpoint and demand/prefix differences. This is not end-to-end learned HRL.
