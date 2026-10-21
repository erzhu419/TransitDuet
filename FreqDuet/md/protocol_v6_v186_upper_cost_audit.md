# V186: read-only V185 upper tails and lower-bound cost

Read only V185's consumed 32 days/12 actors/two modes. Reproduce all saved
path endpoints and support counts before reporting diagnostics; no new replay,
training, interval fitting, map update or main-policy change.

Hour2, clearance, hour13 and nested hour13 moving-launch: retain original
physical-episode/query weights, cross actor/shared-day bootstrap5000,seed20260803.
Upper is exactly V172: report exceedance, excess, headroom and component biases,
actor/day/interaction variation and signed daily component covariance.
Decompose paired score exactly into signed width cost minus lower-tail loss
saving. Describe existing V178 observable strata plus recorded completed-prefix
service sign; missing support remains missing. Do not turn consumed-day profiles
into fitted risk gates. All detailed actor/day/state profiles stay on server.

High-priority manual scheduler dispatch, node001-node006 only; local retrieval
limited to short logs and compact summary. V184/V172/V151/V146, window, tails
and controller remain frozen. Decide the next separate protocol after analysis.

Completed:source9656fd38f4,high/manual t210798,node004,12 workers;5 tests passed,
768 path endpoints/support counts and crossed intervals reproduced. Pulled only
163295-byte summary. Upper-rate day variance share95.39-96.63%,actor<0.46%;
20/32 days exceed10% at hour2,8/32 clearance,9/32 hour13 (both modes).
Day bias covariance:running67-83%,service19-34%,residence small/signed;descriptive,
not causal. Clearance width cost4.965/4.714s,loss saving4.393/4.183s,net+0.573/
0.531s (old/handoff);both net CIs cross0. Lower misses improve9.67% to~7.9%.
Remaining2/3/4 stops:positive score CIs in both modes;6 stops:negative CIs;
known-command also costs more than it saves. Sparse1/8-stop cells have no CI.
These are unadjusted descriptive strata,not a validated gating rule.
Next:keep this candidate unpromoted. Use only original64 calibration days to
check completed running-prefix risk information within existing observable
distance/stage strata,then preregister any new candidate and untouched cohort.
Do not fit or choose a distance cutoff on these32 consumed confirmation days.
