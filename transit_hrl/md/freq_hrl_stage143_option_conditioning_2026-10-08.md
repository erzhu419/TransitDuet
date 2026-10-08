# Stage143: Same-Label Update Conditioning

Stage142 smaller probes reduce even response but do not repair period100 mean
return. Wide/local gradient cosines remain0.893842-0.944620 and noise-fold
agreement stays weak. Upper already receives physical state and tracking error.
Test update conditioning before paying for further labels or expanding seeds.

Roots410037/410049, periods50/100, horizon1200. Replay one mean path per existing
training scene/noise panel and reconstruct both archived wide0.05/local0.005
option gradients. Compare raw392D and the existing causal26D summary(mean,
trend, latest measurement, physical/error state, clocks). Only the incremental
mean fit changes; the original warm actor and strong lower remain unchanged.
Policy std0.05, authority0.05, damping1, RMS0.001767767/KL0.005, both signs.

Both probe targets and pooled/A/B masked label panels share one factorization
per representation. Record preconditioned panel cosine, training linearized
gain by panel and raw/compact functional mean-step cosine. Evaluate32 fresh
paired scenes/period under mean and sampled upper. Primary wide-compact-plus
versus wide-raw-plus; local-probe comparison is the prespecified companion.
Report warm/forecast, both signs and tracking reduction; no evaluation winner.

Per root32replay+1,216evaluation=1,248episodes/1,497,600steps, zero new queries;
64forecast aliases and four shared factorizations. Two roots2,496episodes/
2,995,200steps. Prior query costs stay separate. Sixteen workers+parent/12GiB,
dynamic node001-node006, compact JSON only, no aggregator or checkpoint writes.

## Limitations

The26D projection restricts the update, not the warm actor's original input or
a new frequency encoder. RMS matches on training states, not decoded control.
Noise panels share the training design and scenarios;
their diagnostic is not held-out scene validation. Two-root development only,
not a confirmation CI, equal-cost full HRL training or frequency-specific proof.
