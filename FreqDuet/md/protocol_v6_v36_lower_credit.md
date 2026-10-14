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

Implementation `489c22ce02` is frozen in the server-side `source/` directory.
Regression `t152858` passed five tests. The initial smoke caught a deployment
checkpoint being passed to the full-training-state loader; the diagnostic now
uses the existing deployment loader, leaving all historical files untouched.
Replacement smokes `t152859-t152864` passed in all three configs and both actors:
54 rollouts, unchanged V35 controls/repeats, and all seven action bins
(0, 5, 10, 15, 20, 30, 45 s) passed exact-prefix and same-action checks.
Formal cases `t152865-t153152` are released after these six smoke ready markers:
864 contexts and 6,624 complete rollouts, including references/repeats.
Aggregation `t153153` waits for all 288 cases. Placement is soft-preferred from the live
CPU plan across node001/node002/node004/node005/node006, with only node001-node006
eligible. Summaries include action terms and measured action consequences
relative to zero hold, plus paired early/late action-preference contrasts.
