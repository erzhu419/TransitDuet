# V190: supported leg innovations and full-ETA tail risk

Original64 days2951001-2951064 only;12 actors,both modes. Freeze V151,V146,
V172,3600s window,.1/.1 tails and main. Seal protocol before risk associations.
No V185/other confirmation outcomes,new days,replay,training,point or interval fit.

Reuse V189 road/entry-hour alignment,strict pre-entry native mean and arrival
availability. Two fixed signals,scaled by first-leg remaining fraction:matched
duration mean minus current native mean;RMS(duration minus pre-entry mean).
The latter is surprise magnitude,not sample standard deviation;one completed
traversal is defined,not zero variance. No point overlay or parameter search.

Retain native/V172 support and >=9 strict completed global service episodes for
the service-prefix control. Count complete-input days per actor/mode/scope/road/
entry-hour;require19 independently in all64,first32,last32. No pooling or fallback.
V189 nine-traversal availability remains diagnostic,not a new admission threshold.
Reproduce V189 denominators and raw support weights before interpreting results.

Targets:full-ETA lower/upper misses under frozen V172;excesses and interval score
are secondary. Within actor/road/entry-hour centering retains cross-day variation.
Control query clocks,distance/exposure,forecast levels and bound margins,native/
matched support clocks and counts,global service prefix,current stage;also control
the other signal. Standardized covariance projection only,no deployed estimator.
Original episode weights remain;report missing mass rather than reweighting it.

10000 crossed actor/shared-day bootstrap draws,seed20260803. Primary scopes:
clearance and hour13 moving-launch.16 primary correlations(two modes,two scopes,
two signals,two tails);Bonferroni percentile intervals at1-.05/16. Hour2/hour13,
excesses and scores are diagnostic95% intervals. Expected signs:signed lower-,
upper+;RMS both+. A replicated pair needs expected-sign family intervals in both
modes on all64 AND expected-sign95% intervals in both halves/both modes. Missing
support or variation is inconclusive. This is calibration-only linear association,
not independent prediction skill or evidence for changing the controller.

High/manual scheduler,node001-node006;tests and one real archive before1536 paths.
Resample per-key first moments and summed second moments;verify against direct
calculation. Detailed actor/day diagnostics stay server-side;small summary only.

Pre-inspection supplement:also report95% single-signal associations controlling
the same forecast/state variables but not the other innovation. This distinguishes
shared information from unique information;the original primary gate is unchanged.
Seal the supplement before saved-moment analysis,without opening parent risk
results. Reuse only V190 moment cells,write a separate readout,keep19-day masks.
