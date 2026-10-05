# Stage120 Bounded Plan Authority

Stage119 learned upper gain transferred, but only about 0.0007 return on a 1141-1152
baseline after 65.1M training/evaluation steps. Test amplitude and execution authority
before spending more on this fixed-lower branch.

Keep every Stage112 final learned lower, all 392 feedback features, standard deviations,
forecaster, native task and plan clocks frozen. Intervene in one anchored Bernstein
coordinate for one option by +/-0.05 or +/-1.0. Other options execute zero residual.
Compare existing advice execution with a bounded reference-tracking correction:

`mean_new = mean_existing + 0.05*tanh((donor(feedback+plan_delta)-donor(feedback))/0.05)`.

Only the counterfactual donor query changes target error and causal target velocity by
curve-minus-forecast position/velocity. Original physical feedback/history, reward,
noise and main lower input stay intact. This is a new execution interface, not rescaling
tiny learned advice weights or adding a free constant action bias. Zero residual must
reproduce forecast exactly. Record the two extra donor forwards per reference step.

Eight roots, periods 50/100, four fresh queries/root/period at 0/300/600/900. Independent
suffix-noise panels A/B; same scenario/prefix and paired innovations across all variants.
A selects among zero and sixteen signed directions; B scores complete suffix return;
reverse and average. Ten corrected endpoints, equal-root bootstrap65,536. Pre-register
minimum conditional gain **0.5 suffix-return units** at both periods. This is a prospective
screening threshold, not a retrospective significance or application utility claim.

Train the new channel only if large-reference gain CI is above 0.5 and its advantage
over large-advice CI is above zero at both periods. Large-advice alone above 0.5 instead
supports studying amplitude in the existing channel. If all tested gain CI upper bounds
are below 0.5, stop this tested fixed-lower branch. Otherwise report inconclusive, without
automatic seed extension. No algorithm training is authorized by a mechanical preflight.

Budget: 8,576 full episodes / 10,291,200 native steps; four workers/root, five CPUs,
8 GiB RAM, dynamically placed by scheduleurm across node001-node006. Mechanical preflight:
268 episodes / 80,400 steps; no performance-based changes to the frozen full protocol.
Scalar JSON stays server-side except compact summaries; no checkpoint or trace writes.

## Limitations

Crossfit selection uses the future simulated suffix; it is not a deployable causal policy.
This finite candidate test is neither a global upper bound nor proof of frequency HRL,
learned promotion, joint training or equal-compute superiority. The correction is bounded
in Gaussian mean space; curves are world-clipped, not certified collision-free trajectories.
Stage67 critic HOLD and earlier negative results remain unchanged.
