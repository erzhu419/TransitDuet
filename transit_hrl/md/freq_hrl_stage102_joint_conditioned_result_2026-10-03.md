# Stage102 Joint Conditioned Learning: Full Result

All nine tasks t129115-t129123 finished with exit0. Server-side qualification and read-only reaggregation exactly match the official summary. Joint conditioning is supported: all four preregistered primary corrected CI lower bounds are positive.

| Primary endpoint | Mean reward difference | Bonferroni20 corrected CI |
| --- | ---: | --- |
| 50/joint_conditioned_minus_joint_independent | +0.425053 | [0.102134, 0.746989] |
| 50/joint_conditioned_minus_base | +4.881362 | [4.090590, 5.629187] |
| 100/joint_conditioned_minus_joint_independent | +1.872106 | [0.854208, 2.830962] |
| 100/joint_conditioned_minus_base | +8.068784 | [6.691109, 9.112286] |

All four contrasts are positive on every one of eight original new-teacher roots. The unchanged equal-root bootstrap65,536 / Bonferroni20 family contains 14 positive, 4 negative and 2 inconclusive endpoints. No seed, period, checkpoint or short-return admission was used; preflight negatives remain recorded.

The conditioning advantage is lower-driven. Swapping conditioned lower into the independent joint upper gives +0.426612 / +1.878310 at50/100, versus total joint-route differences +0.425053 / +1.872106. On the same conditioned lower, conditioned upper loses to independent upper: -0.001559 CI [-0.002823,-0.000097] at50 and -0.006203 CI [-0.010292,-0.000923] at100. It also loses on the independent lower at both periods. Keep all four negative upper-swap contrasts; this is not positive upper/lower specialization evidence.

Exact cohort cost: 35,840 native episodes /43,008,000 steps, 512 actor-mean updates, 73,728 extra upper-replay forwards and32 registered final checkpoint writes. All actor-specific credit, independent evaluation and std/value/source-Adam/forecaster/decoder freezes passed. Critic fits, fitted forecasters and raw native traces remain zero. Native wall time1040.19-1104.76 s/root; compact JSON75,647 bytes before pretty-printing, no checkpoints or raw evaluation rows pulled.

Next: evaluate frozen Stage102 joint policies against Stage99-lower + Stage100-U0-upper staged policies on fresh paired rollouts. Their method-path samples, actor-update counts, common-upper replay and cumulative nominal call-weighted KL match; the phased staged recipe uses16 single-actor updates versus8 simultaneous joint updates. Do not include extra-compute Stage101 UJ refinement in this equal-budget comparison. Preserve both independent/common controls, all roots, both periods and both teacher baselines.

## Scope
Teacher-initialized same-task joint MC mean learning is supported, not full actor-critic, frequency superiority, unseen-task generalization or upper/lower synergy. Preparation/campaign costs remain separate from method-path costs; the next comparison uses different registered training rosters and paired fresh evaluation, not an isolated proof of update-order causality.
