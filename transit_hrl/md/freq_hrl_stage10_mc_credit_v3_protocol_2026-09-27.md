# Stage-10 Full-Return Credit Development Screen

Date: 2026-09-27

Protocol: `pointmaze_termination_mc_credit_v3_development`

The stochastic diagnostic found frequent but poorly conditioned early calls.
The v1 PPO used GAE lambda 0.95 per **decision**, although each action changes
the number of subsequent checks. Its trace weights therefore depend on the
chosen termination time rather than physical time. This screen changes only
lambda to 1.0: the actor receives the undiscounted return-to-go from external
tracking ISE, matching the episode objective without decision-count decay.

The frozen Stage-10 controller, 37 causal features plus offset, Bernoulli PPO,
normalization, eight branch-fit paths, eight checkpoint-selection paths,
18 training iterations, five deterministic checkpoint evaluations, 16 held-out
paths, 24 upper calls per episode, and 230,400 additional trigger-training and
selection primitive steps per root remain unchanged. Four fixed-seed
stochastic evaluations per held-out path match the diagnostic readout and are
not used for selection.

Preflight root `208001` checks execution and accounting. Development roots
`209011` and `209061` are reused, already-revealed Stage-10 roots. Compare
per-root stochastic and deterministic ISE, return, early calls, and the old
stochastic and fixed baselines. Proceed to a fresh-root protocol only if both
development roots improve stochastic ISE over the old policy and fixed plan;
otherwise retain the negative result and turn to direct counterfactual action
credit. No CI or paper claim is attached to this screen.

Tasks use scheduleurm dynamic placement on node001-node006, one CPU and
1536 MB each. Only compact `result.json` is synchronized.
