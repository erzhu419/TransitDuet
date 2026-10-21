# Stage304: Closed-Loop Result And Historical Paper Evidence

The registered closed-loop cost/wait primary fails. Historical positive results
remain real, but do not jointly establish the original domain-general Freq-HRL
claim. No new scheduler tasks, training or simulation were launched this turn.

## Completed experiment

Run `native_transit_closed_loop_confirmation_stage304_20261021_r2`:
t208544-t209823, all 1,280 tasks done/exit0, zero retries. Existing-result
aggregation on node002 qualified 219,520 native episodes including inner menus
and identities. Only 36,252 report bytes and a 12,799-byte scalar audit were
pulled; raw data and models remain remote. The completion receipt records this
direct single-CPU aggregation separately from scheduler execution.

The independent unit is one complete scene/episode: 320 scenes, 32 in each of
ten root/regime strata, with four paired controllers. Profile0 and the lower
are fixed; only upper decisions 8/9 use the tested rule. These are neither
1,280 independent worlds nor independent optimizer seeds. Both intervals below
measure full-episode sampling: stratified bootstrap5000/seed304 and paired
stratified normal95. Neither is a fixed-prefix conditional-future MC interval.

## Frozen results

All numbers in the next three tables are multiplied by 1e6; waiting uses
minutes. Negative favors the first controller. Cost is the registered composite
episode cost, not monetary operating expense.

| Comparison / metric | Mean | Episode bootstrap CI95 | Paired normal CI95 |
| --- | ---: | --- | --- |
| service-only - raw / cost | 1901.331 | [-2729.400, 6937.962] | [-2856.632, 6659.294] |
| service-only - raw / fleet | 2604.167 | [-2604.167, 7812.500] | [-2458.577, 7666.910] |
| service-only - raw / service | -702.835 | [-1497.065, -120.823] | [-1414.375, 8.704] |
| service-only - raw / wait | -5706.250 | [-12519.141, -787.422] | [-11818.256, 405.756] |
| base - raw / cost | 1830.591 | [-2798.805, 6863.546] | [-2927.946, 6589.127] |
| base - raw / wait | -6375.000 | [-13334.844, -1390.625] | [-12534.728, -215.272] |
| joint - service-only / cost | 28.572 | [-5.247, 84.056] | [-19.407, 76.551] |
| joint - service-only / wait | 362.500 | [0.000, 953.125] | [-150.046, 875.046] |

| Half / comparison / metric | Mean | Episode bootstrap CI95 | Paired normal CI95 |
| --- | ---: | --- | --- |
| first16 / service-only - raw / cost | 4109.200 | [-5169.012, 13394.129] | [-5334.110, 13552.510] |
| first16 / service-only - raw / wait | -9462.500 | [-22676.094, -212.500] | [-21172.839, 2247.839] |
| last16 / service-only - raw / cost | -306.537 | [-726.662, 25.965] | [-693.570, 80.495] |
| last16 / service-only - raw / wait | -1950.000 | [-5181.563, 643.750] | [-4916.513, 1016.513] |
| first16 / joint - service-only / cost | 56.794 | [-10.494, 164.056] | [-39.168, 152.755] |
| last16 / joint - service-only / cost | 0.350 | [0.000, 1.050] | [-0.336, 1.036] |

Service-only retains 89.51% / 93.57% / 73.93% of base waiting gain pooled /
first16 / last16; joint retains 83.82% / 85.97% / 75.59%. All point ratios
exceed 50%, but the paired waiting-floor inference does not pass:

| Part / margin: wait(service-only) - 0.5 wait(base), both relative raw | Mean | Episode bootstrap CI95 | Paired normal CI95 |
| --- | ---: | --- | --- |
| pooled | -2518.750 | [-5931.289, -13.984] | [-5607.048, 569.548] |
| first16 | -4406.250 | [-11068.906, 268.828] | [-10293.352, 1480.852] |
| last16 | -631.250 | [-2334.453, 853.203] | [-2243.888, 981.388] |

Joint's floor intervals also cross zero. Both transfer gates and the critic
incremental-cost gate are false. Service-only waiting improves 0.342 seconds
(0.0423%) at the point estimate; cost increases 0.0869%. This is inconclusive
cost harm, not established cost improvement, noninferiority or a dual win.

## What happened

Service-only changes outcomes in 21/320 scenes: cost improves in 14 and worsens
in seven; wait separately improves in 14 and worsens in seven (not necessarily
the same scenes). Fleet changes in four scenes, all in first16: three peaks rise
14 to 15 and one falls 15 to 14. Their net mean fleet penalty +0.002604167
exceeds the service saving -0.000702835. The largest waiting-saving scene
(401/high_noise/replica14) supplies 46.66% of net wait gain but increases cost
0.318935; the top three supply 75.96%. These concentration figures are descriptive.

Joint and service-only have identical choices and outcomes in 315/320 scenes;
their peak fleet is identical in all 320. Thus this risk filter did not alter
any of the four consequential fleet events. Joint instead loses service at the
point estimate. Against full base, joint significantly worsens both cost and
wait under both pooled intervals; service-only significantly worsens cost.
There is no indication of missing outcomes or broken accounting in this run.
Stage301-to-304 also changes profile mix, decision positions, on-policy states
and evaluation population; the result does not isolate repeated decisions alone
as the causal reason for the loss of confirmation.

## Historical positive evidence worth preserving

| Evidence family | Positive result | Paper interpretation |
| --- | --- | --- |
| MuJoCo v12 / v14.29 | v12 responsibility LF falls 78.2-89.6% with return NI on three tasks; v14.29 accepts 47/48 fresh optimizer anchors | Responsibility restoration under a reward floor; 38/47 acceptances preserve behavior exactly, not control improvement |
| Quant v7.4 | 8/12 Holm-controlled contrasts improve across 24 training replicates | Mixed baseline evidence: recurrent generic HRL has significantly better return; no uniform dominance |
| PointMaze Stage9 | Eight fresh optimizer roots: return +40.662 [31.548, 49.777], ISE reduction 0.506 [0.408, 0.603], exactly 24 upper calls | Strongest independent closed-loop plan-validity trigger confirmation, within one task family |
| PointMaze Stage54-57 | Learned forecaster/teacher clone improve; Stage55 joint beats frozen at both periods | Conditional learnability and execution value; full joint-gain gate fails |
| PointMaze Stage87-100 | Stage88 fresh-sample joint-minus-lower +0.222/+0.480; Stage89 isolates upper contribution; Stage100 confirms staged upper gains on eight new teacher roots | Genuine learned upper/mean-policy value; most route advantage is inherited lower quality, and strong fixed-upper comparisons still fail |
| PointMaze Stage129/131/135 | Stage135 compact upper beats forecast +1.986 [1.490, 2.364] / +1.079 [0.679, 1.334] at periods50/100 | Frozen-lower internal replication, gains only 0.1724%/0.0947%; Stage136 ordinary joint PPO destroys warm-start gains |
| Transit Stage167/168 | Waiting improves about 0.20-0.32 min | Short-term service potential, offset by extra peak fleet cost |
| Transit Stage247/248/253 | Independent futures preserve waiting/service gains of old247; Stage253 old247 pooled cost/wait points -0.000105371/-0.000565023 | Known-prefix single-intervention evidence; Stage287 new-prefix confirmation does not establish the same dual gain |
| Transit Stage281 | Cross-future service information yields roughly 0.0093-0.0099 min less wait with pooled cost savings | Information-value probe using query labels, not a learned deployable policy |
| Transit Stage288 | Fresh refits improve held cost; wide-minus-raw cost -218.29e-6 CI [-342.88,-103.51]e-6 | Conditional cost baseline; wider coverage itself is not better than dense and wait does not improve |
| Transit Stage292/294 | Frozen abstention reproduces waiting gains, including 240 new Stage294 prefixes | Prospective service/wait transfer, not supported joint total-cost improvement |
| Transit Stage296 | Both cost/wait improve versus raw under both CIs on 120 fresh prefixes | Registered primary fails: only 29.50% of base waiting gain retained |
| Transit Stage298/301 | Two separate 400-prefix cohorts confirm joint cost/wait versus raw; Stage301 service-only also improves both CIs and halves | Strongest Transit single-intervention result; no incremental learned fleet-critic benefit; not Stage304 closed-loop confirmation |

Sources: the dated Stage documents named above, authoritative MuJoCo/Quant
source reports, and Stage304's `report/result.json`, `descriptive_audit.json`
and `evaluation_completion_receipt.json`. The original design is
`freq_hrl_dev_manual.md`, sections0/1.4. Early C1-C9 artifacts remain excluded
by the existing registry; positive estimator/proxy results are not new control
evidence. Historical versions, profiles and repeated cached analyses are not
independent replications of one algorithm.

## Paper decision and next step

There are defensible paper conclusions: causal plan-validity information can
improve fixed-budget PointMaze closed-loop control; a frozen service-aware
finite-MC Transit rule can improve one-intervention outcomes on new prefixes;
responsibility-coordinate restoration can preserve reward. These are different
methods and endpoints, not three domains confirming one Freq-HRL algorithm.

The original headline remains unproven: frequency-bound upper/lower learned
control, learned promotion, leakage suppression and performance improvement
have not jointly survived strong matched baselines and independent closed-loop
evaluation. The current evidence does not justify a top-tier domain-general
Freq-HRL performance claim. Lower prediction error is not the missing proof.

Pause computation as requested. Next work is manuscript consolidation: choose
one executable method and its comparator, make Stage9 or the bounded audit
contribution the central argument, and report Transit Stage298/301 together
with Stage304 as a transfer boundary. Keep learned-upper positive results,
failed joint-training controls and computational costs in the same account.
Do not promote the finite-MC rule as an amortized learned policy or splice
different positive mechanisms into a single cross-domain success claim.
