# V92: read-only elapsed-time covariance review

Use only the 192 compressed V91 primitive records already on-server. No new
rollout, policy update, model fit or candidate selection. Preserve the failed
original joint gate and frozen main policy.

Split other elapsed into cumulative nonholding stop elapsed on matched
observed stops and the remainder. A stop interval is departure minus arrival
time, less the increase in executed holding; add it after the corresponding
departure, not to that stop's arrival. The remainder includes movement and
unobserved terminal/clock effects, not necessarily pure road travel.

Expose the full variance and cross-covariance changes, and require them to
reconstruct V91's parent allocations and measured CV. Split first/last service
boundary changes using the same timing components. Quantify CV contributions
from changed versus unchanged actual predecessors, using each path's full-day
centering/normalization; a small changed-edge count is not evidence of small
impact. All decompositions remain descriptive, not causal mediation.

Run focused tests, read/compute every existing record and resample demand days
through scheduler on node001-node006. Retrieve logs and small summaries only.
Stop on inconsistent timing, unmatched completed stops or failed identities.
The resulting service/covariance evidence determines whether a new control
experiment has a concrete mechanism to test.
