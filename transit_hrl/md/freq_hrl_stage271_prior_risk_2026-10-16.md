# Stage271: Prior-Preserving Risk Residuals

t165870/node005: done/0, 84.15s, no retries; 20 tests passed. 6,336 residual
fits and constrained paths; cold/prior models reused, no native episodes.
Pulled 1.11 MB report plus 12.8 KB posthoc scalar sums. Replay errors <7e-16.

Penalty .01, mixed raw-prior bounded residual vs raw: MSE -9.23% on training
states with opposite futures, but +21.04% on excluded prefixes and +4.67% on
held prefixes. Relative to cold MSE, the last two improve 33.59% / 11.31%.
All three bounded prior variants still lose to their own mixed OOF/held priors.
Adding only the prior's training-projection residual changes held cold MSE
by +0.16%; recovering discarded prior information alone does not fix the loss.

Training RMS checks pass, yet raw held/excluded correction RMS reaches
1.38 / 1.58 times the training radius. Scalar loss decomposition separates
overshoot from direction: mixed raw alignment is positive, but mixed_a OOF
alignment is negative in both future directions. Uniform positive shrinkage
cannot improve that subgroup's empirical fixed-direction loss. No multiplier
was selected from these outcomes.

Next Stage272: matched-capacity observable-state-conditioned residual vs linear,
with a fixed query-local output bound and a frozen clipping control. Keep priors,
historical radius and causal inputs; no seed/scenario-ID gates or native rollout.

Limitations: reused-label diagnostics, not seed CI or policy improvement. The
projection is training-support-specific; the RMS cap is not global no-harm.
