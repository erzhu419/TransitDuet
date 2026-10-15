# Stage244: independent-future critic targets and calibration

Freeze raw references; reuse Stage243 same-future models and Stage242 panels.
Four modes cross target A/B with variance/signal calibration A/B: same,
calibration-only, target-only and disjoint. A is the raw training bank; B is
its old unused complement. Both outer directions are included. Each estimate
uses2 inner or4 full draws, same54/60 states,198 features,17 candidates and
the frozen A-MSE total residual RMS cap. No new raw fits or native replay.

OLS on B is analytically recomputed within A's cap. Record cap activity rather
than interpreting a blocked update as absent signal. Preserve Stage243 controls
and OOF rules. Primary: `disjoint_variance_group_pooled_safe`. Independent
Stage241 future4+4/8 and outer held prefixes are evaluation-only. Seal actions
first; report feasible-action MSE, gate regret and separate service/fleet costs.
Shared state/future outcomes count once in conditional sampling SE.

Next: compare calibration-only with same, then disjoint with target-only;
compare target-only with same, then disjoint with calibration-only. Check the
full-refit held-prefix result before accepting an OOF gain.

Limits: target/calibration sample counts match, but distinct raw-plus-critic
future counts differ across modes. This is reused development evidence, not
new confirmation or training-seed CI. Stage169 reserves and defaults unchanged.
