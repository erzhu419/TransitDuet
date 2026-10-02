# Stage92: supported conditional lower-credit improvement

t124963-t124971 all done/exit0; mechanical gate passed. Eight roots,22,528 episodes /27,033,600 steps; wall661-695s/root, peak4538-4831MiB. Pulled106,531-byte compact JSON only; final checkpoints remain server-side.
Lower means were trained with within-pair shared upper innovations and independent lower streams under frozen source U0 or final Stage88 UJ. Evaluation used the unchanged normal RNG path. Controls are Stage90 LM (frozen U0) and Stage91 LC (frozen UJ), with the same lower budget and training rosters.

| Primary: shared versus independent upper-noise training | period50: mean [CI]; positive roots | period100: mean [CI]; positive roots |
| --- | --- | --- |
| Matched frozen U0: source_common minus source_matched | +0.38354 [+0.08202,+0.82545];8/8 | +1.53664 [+0.70029,+2.63077];8/8 |
| Matched frozen UJ: joint_common minus joint_fixed | +0.37938 [+0.08083,+0.81206];7/8 | +1.50076 [+0.67581,+2.53743];8/8 |

All four primary corrected CI lower bounds are positive: the frozen global conditioning-benefit gate is supported. Effects are approximately0.04%/0.18% of the independent controls' dense episode returns, not a large relative-return improvement.
All26 equal-root bootstrap65536 / Bonferroni26 CIs were independently recomputed, together with evaluation-mean contrasts, root/total costs,64-episode updates and freeze metadata;20 positive,2 negative,4 inconclusive. The256 lower updates,32 final checkpoints and147,456 extra replay forwards match preregistration.
The new frozen-UJ lower also improves over Stage88 joint LJ at both periods: +0.37370 CI[+0.07607,+0.80362] and +1.49134 CI[+0.68390,+2.51582]. This is a conditional lower repair, not new joint training.
Initial-update scenario-gradient covariance trace ratios versus controls average0.614 at50 and0.359 at100 for either fixed upper. All16 root/batch comparisons decrease at50;15/16 decrease at100. These diagnostics are descriptive, not a variance-dominance theorem or isolated causal attribution.
Retain negative secondary endpoints: source-U0 evaluation favors source-common over joint-common-trained lower at50 (difference -0.01129, CI[-0.02274,-0.00010] in the registered opposite direction); base-minus-zero at100 remains negative (CI[-1.31086,-0.22679]).
Next: freeze a fresh-training/fresh-evaluation replication with newly trained independent-noise controls at both fixed uppers. Keep the same radius, periods, per-learner sample budget and all-primary decision; do not pool stages or select the stronger period.

## Limitations
Same teachers and intentionally reused training scenarios; this is not independent training replication, joint learned HRL, full actor-critic or frequency superiority. Shared innovations do not fix upper actions and reduce independent upper draws; replay adds measured compute, so this is not equal-total-compute. Stage91's original repair remains unsupported; Stage67 HOLD and earlier negative results remain unchanged.
