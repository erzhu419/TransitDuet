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

## Limitations

Small sampled-state KL does not ensure deployment improvement. This does not
repair critic temporal error or stochastic-upper versus mean-policy mismatch;
those remain open if bounded/projection updates still fail. This is not joint
HRL, frequency responsibility or promotion confirmation. Earlier results stay.
