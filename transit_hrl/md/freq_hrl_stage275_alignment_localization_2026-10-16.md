# Stage275: Alignment Concentration and Future Stability

t165990/node002: done/0, no retries; 14 tests and both cold entrypoints passed.
640 cached heads / 3,840 model-probe rows; zero fits, inference or simulations.
Pulled 741 kB scalars only. Stage274 replay / partition errors <1.05e-17.

Primary clipped RMS head, .01, mixed pooled, common-action pairs:

| Rank prefixes using | Evaluate using | Top3 net share | Top3 absolute share | Remaining alignment gap, 1e-6 |
| --- | --- | ---: | ---: | ---: |
| Assigned16 | Opposite16 | 78.06% | 33.48% | +12.665 |
| Opposite16 | Assigned16 | 90.70% | 37.76% | +5.204 |

Both directions select 401/914010401, 397/913970003, 401/914010303. Contributions
are concentrated but strongly cancel: these shares are not shares of all error.
Probe8/9 contribute +54.479e-6; remaining probes sum to -2.438e-6. Fixed-head/probe
sign agreement is 453/608 (74.51%); sign-flipping rows supply 11.78% of the net gap.
The mixed pooled mismatch is concentrated and partly repeatable, not uniform.

Next Stage276: compare dominant prefixes/probes with the rest using frozen caches;
separate raw-residual scale from normalized correction/residual alignment before
another loss/map change. No prefix/probe-ID gate or default change.

Limitations: paired coverage 31.7%; repeated development exposures, not seed CI
or policy gain. Bank1 alone has different top3 rankings; natural weights also
change rankings. Scheduler end precedes start, so elapsed time is not reported.
