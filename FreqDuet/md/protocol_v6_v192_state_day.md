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
