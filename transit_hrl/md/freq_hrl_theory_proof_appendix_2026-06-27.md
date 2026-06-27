# Freq-HRL Theory And Proof Appendix Skeleton

Date: 2026-06-27

The target is not a universal convergence theorem. The target is a defensible set of sufficient-condition statements that support the method's claim boundaries.

| proof_item | status | statement | paper_use |
| --- | --- | --- | --- |
| causal encoder lemma | formalized_skeleton | If E_phi only consumes x_{<=t}, no policy action can depend on future exogenous observations through the encoder. | guards against lookahead leakage in all frequency features |
| frequency responsibility proposition | formalized_skeleton | Under band-separated exogenous drivers and bounded cross-band covariance, routing LF features to upper and HF residuals to lower reduces cross-level credit variance. | turns frequency decomposition from feature engineering into an HRL responsibility principle |
| leakage bound | formalized_skeleton | With an LPF penalty on cumulative lower actions and an HPF penalty on upper actions, responsibility leakage is bounded by the constraint budget plus optimization residual. | supports the no-tradeoff boundary when performance gates also pass |
| promotion detection tradeoff | formalized_skeleton | Persistence thresholds induce an explicit false-positive/false-negative tradeoff between early replanning and shock overreaction. | explains why promotion claims are stress-registered rather than universal |
| paired-CI claim rule | formalized_skeleton | A claim is supported only when paired deltas pass direction-aware confidence gates over matched seeds or source windows. | connects statistical evidence to claim boundaries |

## Suggested Assumptions

A1. The exogenous process admits a causal approximate band decomposition with bounded reconstruction residual.
A2. The upper action affects low-frequency plan variables more directly than high-frequency residual dynamics.
A3. The lower action affects high-frequency correction more directly than long-horizon plan variables, up to measurable leakage.
A4. Paired experiment seeds or source windows are exchangeable enough for direction-aware CI gates.

## Proof Strategy

1. Prove no-lookahead from encoder causality.
2. Bound cross-level credit variance under band-routed observations.
3. Bound responsibility leakage under upper-HPF and lower-LPF penalties.
4. Derive promotion threshold false-positive and false-negative tradeoffs.
5. Connect empirical paired-CI gates to conservative claim wording.
