# Stage137: Bounded PPO Direction Diagnosis

Stage136 retains warm-start upper gains but its first ordinary upper PPO update
destroys them at all four root-periods. Output RMS rises5.85-8.76times. This
experiment changes update scale and subspace, not the training target.

Roots410037/410049, periods50/100, horizon1200. Reconstruct the exact16 sampled
Stage136 training paths and upper optimizer seed. Check their returns and update
against the completed source; this is replay, not fresh training evidence.
Retain the original Adam displacement. Generate both signs at the inherited
empirical KL radius0.000555556, either in the raw392D parameter space or after
orthogonal projection into the existing causal26D row space. Std0.15, authority
0.05, teacher and forecast are unchanged. All deployed lower/critic weights
remain the warm-start ones; upper critic optimization used to reproduce the
original update is counted but not deployed in the comparison.

Evaluate32 new paired scenes on forecast, unchanged upper, raw Adam, bounded
Adam plus/minus and compact Adam plus/minus. Record exact training-state KL,
mean-step RMS, reward and squared tracking-error integral. Positive tracking
reduction means lower error. No native-return fit, winner selection, confirmation
CI or automatic extension. Smaller losses alone are not improvement; a reverse
direction win would suggest credit/objective error, not validate a reversed loss.

New cost/root480episodes/576,000steps; total960episodes/1,152,000steps. Eight
workers+parent/8GiB, scheduler dynamic node001-node006. Compact JSON only, no
checkpoint or trace writes. Inherited Stage136/135/source-chain costs separate.

## Run Receipt

Implementation `54683cd906`, registration `68ee62bb39`; seven related tests
passed, including exact update replay and reduced runner budget. Both Stage136
source cells and four Stage135 selected upper files are available on node004.
Run `pointmaze_bounded_ppo_stage137_probe_20261008_r1`: `t136920` root410037,
`t136921` root410049 are DONE on node004/node006.93,561bytes of compact JSON
were fetched, without checkpoints or raw trajectories.

## Result And Next Step

Both cells pass replay, freeze and exact accounting:960episodes/1,152,000steps.
Raw Adam KL0.100270-0.152228 is180.49-274.01times the fixed radius. Bounding
reduces damage but does not establish an improving direction:

| Period | Raw Adam - warm | Bounded plus - warm | Compact plus - warm | Bounded plus - minus |
| --- | ---: | ---: | ---: | ---: |
| 50 | -3.001409 | -0.040176 | -0.040452 | 0.153597 |
| 100 | -3.194713 | -0.015675 | -0.016252 | 0.023302 |

Forward bounded/compact gains are positive in two of four root-periods. Both
reverse averages are negative. Projection discards44.7%-48.6% of weight-step
energy but has almost identical deployed results; discarded parameter energy
does not establish relevant control leakage. Tracking reductions are mixed,
and reward/tracking signs need not agree because reward is nonlinear.

Stage138 changes actor credit alone: replace one sampled option by its current
mean at the identical prefix, replay the stochastic continuation with the same
innovations, and use the paired full-return difference. Keep std0.15 and the
fixed KL radius, shared on-policy batch, critic targets and frozen lower.
Fresh training/evaluation scenes, both update signs, no evaluation winner or
confirmation CI. The stochastic-training/mean-deployment mismatch remains open.

## Limitations

Small sampled-state KL does not ensure deployment improvement. This does not
repair critic temporal error or stochastic-upper versus mean-policy mismatch;
those remain open if bounded/projection updates still fail. This is not joint
HRL, frequency responsibility or promotion confirmation. Earlier results stay.
