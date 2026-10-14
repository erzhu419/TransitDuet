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

2026-10-15 aggregation recovery: all 288 cases completed with exit code zero,
but `t153153` awaited local copies of ready markers (76/288 present while all
288 existed remotely). A small marker-only rsync cleared this prerequisite
backlog without rerunning evaluations. The released aggregator then exposed a
missing import path for the frozen V9 statistics module. Fix `8ac7a21165`
loads that module from `MODEL_ROOT`; the original rollout source is unchanged.
Replacement `t153188` uses a separate `aggregation_source/8ac7a21165` directory
and records rollout and aggregation commits separately. It completed on node004
with exit code zero, aggregating all 288 cases, 864 contexts and 6,624 rollouts.
This closes aggregation execution; interpretation of the action-credit results
and any algorithmic repair remain separate work.

## Initial interpretation, 2026-10-15

The five summary files (148,833 bytes) were retrieved; action-level records and
checkpoints remain on the server. The main configuration gives the following
late-minus-early preferences on identical causal states. Actor ages below count
completed episodes, rather than zero-based checkpoint indices. Intervals are
the existing crossed-bootstrap 95% intervals, not new confirmation tests.

| Continuation actor | Actor action difference, s | Energy-minimizing action difference, s | Journey difference from actor choice, min |
| --- | --- | --- | --- |
| 40 | +7.0486 [2.9167, 11.4583] | +5.6944 [1.5625, 9.9306] | +0.002659 [-0.003060, 0.012727] |
| 200 | +3.8542 [1.3542, 6.7708] | +4.5139 [1.3194, 7.8472] | +0.003676 [-0.006101, 0.017728] |

The late actor and the late Q/pessimism/cost score both favor longer actions
relative to their early counterparts. This does not isolate an actor-only or
temperature-only failure. However, actor/energy agreement on individual states
cannot be inferred from these contrasts or from averaged action-score curves.

The main single-action actor contrasts do not establish a directional daily
journey or CV effect under either continuation actor. This sparse, one-action
intervention is different from V35's full-day lower-policy replacement; its
effect must not be multiplied by the number of daily decisions.

Longer holding is not uniformly better even for the recorded local objective:
45 s versus zero reduces main deterministic remaining-trip reward by
0.276555 [0.127881, 0.456611] under continuation 40 and
0.187991 [0.056148, 0.331582] under continuation 200. These are deployment
reward returns, not entropy-regularized Q targets or the full actor objective.

The GPT6 assessment's resource/credit mismatch is a strong working hypothesis,
not a root cause established by V36. Next use the existing server-side records
for per-state actor-versus-energy regret and Boltzmann-distribution agreement,
then match continuation, entropy and terminal semantics before judging critic
calibration. The frozen runner passes reward_scale=1.0, so reward scaling alone
is not an established mismatch here. A small joint-action feasibility study
should precede any slack/MPC redesign or further large training campaign.
