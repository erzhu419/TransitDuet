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
