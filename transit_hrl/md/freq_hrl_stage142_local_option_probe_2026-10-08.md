# Stage142: Local Native Option Probes

Stage141 mean-query improves on stochastic credit in all four mean root-periods,
but period100 remains negative in equal-root return and root410049. Noise-fold
cosines are0.119442/0.036449 at100. Roughly96-97% of paired query even responses
are negative there. Test probe radius while holding policy noise and step fixed.

Roots410037/410049, periods50/100, horizon1200. Reconstruct wide0.05 probe labels
from Stage141 by exact mean-path replay. On identical training scenes, mean
paths and probe innovations, query local0.005 single-option perturbations;
all other upper options use their conditional means. Policy std stays0.05,
authority0.05, damped mean geometry1, RMS step0.001767767/KL0.005. Both signs;
all lower/values/teacher/forecaster and source weights remain frozen.

Evaluate32 new scenes/period under mean and sampled upper. Primary local-plus
versus wide-plus return, plus both methods versus warm/forecast and tracking
reduction. Record query even/odd response, wide-to-local gradient cosine and
noise-fold agreement. Narrow probes improving return and direction agreement
would favor smoothing bias; unchanged direction/failure favors estimator or
state-conditioning work. No automatic further radius sweep or seed expansion.

Per root32replay+32collection+1,152queries+704evaluation=1,920episodes,
2,304,000steps;576new antithetic pairs and64forecast aliases. Two roots
3,840episodes/4,608,000steps. Prior queries remain separately accounted.
Sixteen workers+parent/12GiB, dynamic node001-node006; compact JSON only.

## Run Receipt

Implementation/results commitbc877d1a5f; preregistration commit746057d7ae,
pushed before submission. Fifteen related focused tests passed. Metadata reads
confirmed both Stage141 source cells and all four selected warm upper files.
Run: pointmaze_local_option_probe_stage142_probe_20261008_r1.
t137081/root410037 and t137082/root410049 were queued at receipt time, with
no assigned node. Preregistration and compact dispatch receipt are retained.

## Completed Result

t137081 finished on node004; t137082 on node006. Pulled248,367bytes of compact
JSON only. Exact contract, budgets, seeds, paired means and update RMS pass.
Total3,840episodes/4,608,000steps; no checkpoint or trace pulled.

| Period | Mean wide increment | Mean local increment | Sampled wide | Sampled local |
| --- | ---: | ---: | ---: | ---: |
| 50 | +0.116027 | +0.146525 | +0.054345 | +0.050964 |
| 100 | -0.114989 | -0.100711 | +0.017012 | +0.013702 |

Local improves mean over wide on average, but only two of four root-periods;
both period100 local mean increments are negative(-0.001039/-0.200382).
Query-gradient wide/local cosines are0.893842-0.944620. At100, even-response
RMS falls0.316133/0.331456 to0.020071/0.020064, while local noise-fold cosines
remain0.112662/0.123746. Curvature is reduced without stable learning transfer;
do not adopt local probes as a confirmed winner. Next: same-label raw392D versus
existing causal26D update conditioning, both probe caches, no new queries.

## Limitations

Negative even response alone is not evidence that the central gradient is biased;
quadratic curvature cancels in the paired difference. One probe direction per
option/noise fold remains noisy. This is a finite-probe development comparison,
not equal new-query budgets, a confirmation CI or joint HRL validation.
