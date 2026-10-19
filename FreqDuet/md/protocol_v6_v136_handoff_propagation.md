# V136: read-only V135 propagation audit

Read192 V135 pairs:160 held-actor primary,32 secondary. Scheduler t195254 and
t195255 exited0 on node006; six accounting tests pass. All saved CV2, holding,
passenger, return and inclusive-delay endpoints reconcile. Source32d3431e98;
no rollout, training, refit, gate change or policy switch. Initial t195253 lacked
existing turnover helper files and was stopped before analysis. Only small
summaries fetched (final164487 bytes); per-stop/trip details stay on the server.

Primary:251 direct cuts,15 increases,69 unchanged requests. Same-state request
overrides average-37.125 vehicle-s per full-day path; realized path holding
changes-26.0 (CI includes0). Handoff-key changes-34.938; other stops+8.938.
Among48 passenger-harm cells, direct requests average-34.792 but other stops
add65.104; net realized holding+30.938. A different case (12071/2351007) cuts
195s holding yet adds18779 waiting person-s, outweighing6576 onboard person-s
saved. The original two actors on new days also fail to repeat the joint gain.

Actual CV2 mean change+51.3222e-6 =diagonal terms-143.4741e-6 +cross terms
193.8239e-6 +normalization0.9724e-6. Covariation changes, not normalization or
larger component variances alone, explain this accounting.26 no-direct-action
cells have identical complete timing. V135's original cell definition remains
34/160; campaign confidence-interval gates still fail.

Limitations: outcome-conditioned groups and covariance/turnover allocations
are descriptive identities, not per-action causal values. Sparse traces lack
complete same-follower ready truth and passenger-level holding exposure.

Next: targeted passive labeling on already consumed paired paths, reproducing
original actions/endpoints before reading same-vehicle ready labels, bilateral
gaps and passenger exposure. Distinguish ETA error from later holding/dispatch
feedback; do not fit a new controller or retune on V135 confirmation days.
