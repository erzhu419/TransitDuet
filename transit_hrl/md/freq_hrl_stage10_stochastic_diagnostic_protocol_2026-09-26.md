# Stage-10 Stochastic-Termination Diagnostic

Date: 2026-09-26

Protocol: `pointmaze_termination_stochastic_v2_diagnostic`

The failed Stage-10 v1 deterministic actor usually waited until the forced
deadline. This **post-failure development diagnostic** tests whether the
policy learned useful stochastic choices that its `p >= 0.5` deployment rule
discarded. It cannot repair the v1 result or serve as independent confirmation.

Training, controller, inputs, PPO updates, checkpoint selection, task, and
planning budget remain unchanged at algorithm revision
`152ca283728352e02ea1017df33cd678fb3bd5e7`. At each old held-out path,
the selected actor is also evaluated under four fixed-seed Bernoulli draws.
Each rollout still makes exactly one upper call per 50-step bin. The extra
rollouts are evaluation only, not trigger training or selection.

Software preflight uses root `208001`; diagnostic roots are `209011` and
`209061`, both already revealed by Stage-10 v1. The registered readout is
per-root mean ISE, return, early calls, and offset distributions for stochastic
versus deterministic termination, plus the stored fixed and Stage-9 candidate
rows. No superiority gate or paper claim is attached to these two roots.
If stochastic termination is active and competent while deterministic is not,
deployment calibration is the next isolated repair. If both fail, the
semi-Markov policy-credit and training objective need reconsideration.

Tasks use scheduler dynamic placement on node001-node006, one CPU and 1536 MB
each. Only compact `result.json` is synchronized.
