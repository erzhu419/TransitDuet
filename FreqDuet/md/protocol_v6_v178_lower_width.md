# V178: read-only V177 lower-tail and width accounting

Read only the 1,536 saved V177 query/bound/joint-label streams. Reproduce the
four target endpoints and crossed intervals before interpretation. No new
rollouts, fits, maps, training or policy changes; all 64 days remain consumed.

Partition by recorded stage/horizon, position, direction/stop, calibration-day
support, native support/survivors/age, clearance elapsed time, remaining stops
and stage x horizon. Bins are diagnostic descriptions, never new interval keys
or gates. Late-sampled load stays unavailable. Preserve original episode and
actor/day weights: subgroup contributions must sum exactly to each endpoint.
Absent subgroups have zero mass; missing finite labels fail, not become zeros.

Separate lower-miss running/service/residence errors from frozen qhigh changes;
report actor/day/interaction variation, day-component covariance and profiles.
For upper widening versus V172, account for width minus saved upper penalty,
including beneficial and costly contexts. Use the existing 5,000-draw crossed
bootstrap; unsupported conditional intervals stay null. Descriptive associations
are not causal effects or fitted deployment gates. Run six four-bucket shards
and aggregation high/manual on node001-node006; retrieve only tiny summaries.

Completed: t210633 passed six tests and saved-path preflight; t210634-t210639
read all 24 actor/mode buckets on node004/node005. t210640 reproduced all four
target endpoints and their crossed intervals. Source da3857a784; summary 664,527
bytes, six receipts 661 bytes. Initial t210632 stopped at the test import before
reading archives; the test path was corrected without changing any frozen code.

Clearance lower-rate variance is 93.31%/93.18% day-associated and only
0.17%/0.19% actor-associated (old/handoff). Daily total-bias covariance fractions
are running 65.96%/66.34%, service 32.47%/32.64%, residence 1.57%/1.02%.
These describe different quantities: residence still has strong state dependence.
Remaining five stops has lower misses 15.95%/15.96%, six stops 30.43%/30.61%;
mean residence bias rises to about +13.5s/+33s, while qhigh stays near 57s/58s.

Envelope score deltas versus V172, seconds, old/handoff:
- Clearance, 19-31 calibration days: +14.21/+15.57; 48-64 days: +3.75/+4.07.
- Moving-launch, 19-31 days: +10.30/+11.54; 48-64 days: +3.76/+4.07.
- Clearance service stage: +17.38/+23.78, but only about 2% scored mass.
- Clearance moving-launch: +3.95/+4.27, about 92% mass, the main cost contributor.
All listed paired 95% intervals are positive. Raising minimum day support alone
cannot explain away the loss. Remaining 6-7-stop apparent score benefits have
wide intervals crossing zero; eight-stop intervals lack resampling support.

Cached-cell decomposition t210642 (source c1d9afa003, two tests passed, summary
20,091 bytes) separates state mix from conditional-rate changes. Remaining-stop
mix accounts for only 5.93%/6.23% of clearance day variance; within-bin changes
account for 94.07%/93.77%. Stage and support-day mix contributions are near zero.
These are separate marginal decompositions, not a joint all-state adjustment.

Next: only the original 64 calibration days should be used to replicate these
profiles and examine within-bin residual variation against query-observable,
strictly completed-prefix information. Establish support and repeatability
before preregistering any new conditional interval. No automatic widening,
support-threshold change, refitting on V177 or controller promotion.

Limitations: these are consumed-cohort descriptive profiles, not causal effects
or deployment gates. Residence is a combined component, not pure holding.
All target point forecasts are <=900s; longer-horizon comparisons are unavailable.
Lower-support/unseen states remain abstentions and conditional statistics null.
