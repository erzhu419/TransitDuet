# V194: signed running news and clocked component mechanisms

Original64 calibration days2951001-2951064 only. Freeze V151,V146,V172,3600s
window,.1/.1 tails,actors/critic/dual and main. No RMS threshold or interval
change,new confirmation,replay or training. These are reused development days.

Hypotheses,registered before inspecting new component associations:
- H1:query-time signed UNABSORBED same-road news J predicts the first remaining
  road's running underestimation within day,not merely day-level demand variation.
- H2:the correspondence can attenuate on subsequent roads;first-road information
  must not be extrapolated to an entire remaining journey without evidence.
- H3:service and post-ready errors can offset running covariance,weakening full
  ETA association even if the first-road correspondence is positive.

Keep I=mean(duration-pre-entry native mean),A=current native mean-mean pre-entry
native mean,and J=I-A,each scaled by the query's first-road remaining fraction.
I,A,J are an accounting identity,not three independent predictors. Native inputs
precede source entry;source arrival is strictly before query. Actual future entry
times never define feature keys. No load observed after query is backfilled.

Use exact V191 common actor/mode/scope/road/entry-hour keys with19 complete-input
days in EACH half. Reproduce V190 query/episode weights and moments before using
new labels;never renormalize a selected episode. Preserve V191 support balancing.
Clip truth intervals to query-to-target-ready. First running completes at first
arrival;all running/holding at target arrival;service/full at target ready. Verify
same physical follower and component sums against frozen V166 labels.

Partition full underprediction(truth-prediction) into first/later running,service,
holding OUTSIDE a current inseparable joint segment,execution wait,and current
joint residence. Never relabel joint residence as pure holding.

Stage execution:seal protocol,run tests and four fixed clock preflights(first
actor,first/last original day,both modes),then prepare/admit all1536 old paths.
Save per-key/day joint moments server-side;this stage does NOT estimate new
mechanism associations. High/manual scheduler,node001-node006;small summary only.

Registered next readout:primary within-key/day covariance,between-day separately.
Fixed original23 query controls plus RMS as a control only;use the same covariate
projection operator for J and all additive outcomes. Primary contrasts:cov(J,first
running)>0 and cov(J,service+holding outside joint+execution wait+current joint)<0.
Full-minus-first also includes later running and cannot alone identify service/
holding cancellation. Later-road covariance describes a summed path,not a per-road
effect or an equivalence test. Eight all64 family contrasts
(two modes,two primary scopes,two contrasts),Bonferroni alpha=.05/8. Require both
mode family signs and both fixed-half95% signs.10000 crossed actor/day draws,
seed20260803;shared actors,independent half days,recomputed projections,fixed
masks/balance. Later-road contrast and component allocations are diagnostic.
Hour13 overlaps clearance and is not independent replication. No first-road
replication:no propagation correction;no offset contrast:no cancellation claim.
Positive calibration association alone is not held-future skill or causal mediation.
