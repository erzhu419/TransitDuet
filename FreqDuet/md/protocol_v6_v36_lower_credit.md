# V36: Lower action values and system-cost credit

V35 attributes the shared holding/journey growth to the lower actor. V36 keeps
the verified optimized V9 runtime, B200 background and upper actor fixed. No
training or policy promotion is performed.

Design: 3 configurations x 12 training seeds x 4 reused scenarios
(45007, 45013, 934011, 934029) x 2 lower continuation actors (40/200), or 288
CPU jobs. Each job selects the first causally observed nonterminal decision
after 2, 6 and 10 simulated hours, without selecting by outcomes. At each shared
state, both historical learners expose their categorical probabilities,
Q mean/uncertainty, cost penalty, entropy, and exact actor-loss integrand.
Immediate APC holding and regularity terms are also recorded.

Every candidate in the existing action alphabet is executed once, then the
unchanged deterministic policy continues to the episode end. Each branch
replays the full simulator/controller from the same checkpoint/scenario, with
exact lower/upper prefix, causal state and RNG comparisons at intervention.
An unchanged repeat and each same-action branch must reproduce the reference;
the reference must reproduce V35. Per-target discounted reward/cost returns
follow the affected bus/trip to its terminal transition. Global outcomes cover
waiting/onboard time, holding, fleet readiness and upper backlog credit.

Interpretation: compare the action preferred by the late versus early learner
on identical states, under both continuation actors. Separate the reward-Q,
pessimism, cost and entropy terms before choosing a repair. Deterministic-trip
returns are deployment diagnostics, not unbiased estimates of stochastic
entropy-regularized Q. Reused seeds and sparse decision contexts do not confirm
a repaired policy or replace the failed long-training regularity gate.

All regression tests, smokes, rollouts and aggregation run through scheduleurm
on node001-node006. Only summaries return locally; checkpoints and action-level
records remain on shared server storage. Formal jobs wait for smoke completion.
