# Stage101 Upper Refinement: Full Result

All nine tasks t129050-t129058 finished with exit0. Server-side read-only qualification and reaggregation exactly matched the official summary. Both upper refinements improve over their fixed-UJ baseline with the same frozen lower; all four primary contrasts are positive on every one of eight roots.

| Primary endpoint | Mean reward difference | Bonferroni28 corrected CI |
| --- | ---: | --- |
| 50/refined_common_minus_fixed_joint_common | +0.338457 | [0.267461, 0.460168] |
| 50/refined_independent_minus_fixed_joint_independent | +0.348376 | [0.277016, 0.464531] |
| 100/refined_common_minus_fixed_joint_common | +0.536197 | [0.459432, 0.643351] |
| 100/refined_independent_minus_fixed_joint_independent | +0.600969 | [0.501127, 0.711296] |

Refinement is supported; matched-upper specialization is not. On the same common lower, common-trained upper loses to crossed independent-trained upper: -0.004276 CI [-0.008679,-0.001136] at50; -0.020367 CI [-0.035861,-0.006260] at100. It also loses under the independent lower at both periods. Total route superiority is inconclusive at50 and positive at100; neither substitutes for specialization. Keep all 6 negative endpoints, including base-minus-zero at both periods. All28 contrasts: 17 positive, 6 negative, 5 inconclusive, with unchanged equal-root bootstrap65,536 / Bonferroni28 and no pooling.

Incremental cost: 22,016 episodes / 26,419,200 steps, 256 upper-mean updates, 48 donor loads, 32 UJ initialization checks and 32 registered final checkpoint writes. Native runtime: 615.06-643.19 s/root. Frozen lower/std/value/Adam/source/forecaster/decoder and independent-noise checks passed; no critic fits, upper replay or raw native traces. Local compact JSON is 52951 bytes before pretty-printing; checkpoints and raw evaluation rows remain server-only.

Next: test joint mean learning from the original Stage96 teachers with separate actor-credit rollouts. Upper credit must retain independent upper/lower pairs; only the lower-credit pair shares upper innovations. Compare conditioned and independent joint learners with equal per-actor samples and call-weighted KL, fresh roles and crossed-actor controls. No Stage101 donor selection or retuning to rescue specialization.

Scope: same-task, teacher-initialized conditional MC refinement with additional compute, not full actor-critic, frequency superiority or unseen-task generalization. Stage100 negatives and Stage67 critic HOLD remain recorded.
