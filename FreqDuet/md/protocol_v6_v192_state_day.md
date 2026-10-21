# V192: common-key observable composition and day residual covariance

Original64 calibration days only. Read saved V190 moment cells; use precisely
the V191 common keys with19 complete-input days in EACH half and its support-only
balanced masses. Reproduce key identities,day counts,masses and both signals'
balanced associations before interpreting any new decomposition.

Seal this diagnostic before reading new results. Focus clearance,hour13 and
hour13 moving-launch,both modes. Keep all V190/V191 gates and negative results.
Compare original query-state mean shifts within this fixed key mixture. Scale
shifts by the average-half within-key SD;this is not nonlinear state matching.

For RMS innovation versus the frozen V172 upper miss,use one reference linear
residualizer from the average of the two half covariance matrices and one
reference residual-SD denominator. Split covariance into within-day and between-
day terms. Decompose each into raw covariance and the two adjustment terms plus
their cross-product;allocate cross-control terms symmetrically among fixed
clock/geometry,forecast/bounds,leg-support,completed-service,stage and signed-
innovation groups. These are descriptive coordinate-dependent contributions,
not causal shares. Retain all days;never select days for fitting.
Also report reference-residual correlations with each half's own residual SD,
and their differences from V191;the additive common-denominator contributions
are not half-specific correlations. Keep the complete cross-control pair matrix
server-side,including off-diagonal terms.

10000 crossed actor/day bootstrap draws,seed20260803. Share actor draws,but draw
days independently within each32-day half. Recompute the reference residualizer
per draw;freeze only the admitted keys and support-mass balancing factors.
Report direct half differences with diagnostic95% intervals,not significance
differences. Detailed day/actor contributions stay on the server.

Freeze V151,V146,V172,3600s window,.1/.1 tails and main. No interval/predictor fit,
new confirmation,replay or training. High/manual scheduler,node001-node006 only;
server tests precede analysis,local fetches contain small summaries/logs only.

Limits:moments describe linear means/covariances,not full within-key state
distributions. Statistically adjusted innovation/event residuals are not physical
ETA component errors. No diagnostic here establishes causal drift or control skill.

## Results and next step

t210845/source9a11be5ab6 completed high/manual on node004,exit0,no retries;7 tests
passed and all6 cases reproduced V191 keys/support/points. Fetched258815 bytes of
summary only. t210844 stopped before analysis:its synthetic control exactly
determined the tested signal;the corrected fixture retains independent noise.
No analysis formula or scientific threshold changed.

Shared-coefficient correlations remain weak in first32. Old clearance:first0.03035,
last0.06976,delta0.03941[-0.02994,0.10639];moving-launch:first0.04309,last0.06653,
delta0.02344[-0.04801,0.09317]. Handoff agrees. Changing the residualizer raises
first-half r by only0.0015-0.0026;it does not resolve the failed replication.

With the COMMON reference denominator,old clearance within-day contributions are
-0.00188/+0.01686(first/last),between-day+0.03276/+0.05144. Between-day positive/
negative sums are+0.05853/-0.02576 versus+0.06140/-0.00996. In moving-launch,
within-day is+0.00290/+0.02001,between-day+0.04070/+0.04550. These contributions
are not half-specific correlations. All direct component-change intervals cross0.

Old clearance's command-stage share falls8.41% to6.78%;its allocated adjustment
is small,and moving-launch has fixed stage yet retains weak first-half evidence.
Most state mean shifts are small relative to within-key SD;service-prefix shift
is0.176 SD but its interval crosses0. Clock/geometry and forecast/bound covariance
adjustments substantially oppose each other;their between-half changes remain
uncertain. Mean shifts alone cannot exclude nonlinear state-composition effects.

Decision:no interval fit,new confirmation,replay,training or main change. The
first-half weakness combines little resolved within-day association with noisy
day-level contributions,not a demonstrated risk sign reversal. Next only inspect
within-day innovation variance,upper-event concentration and day-specific direction
on the same admitted moment cells;do not select days or tune a correction.
