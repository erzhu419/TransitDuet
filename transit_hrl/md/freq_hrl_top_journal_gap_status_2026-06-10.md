# Freq-HRL Top-Journal Gap Status, 2026-06-10

This note updates the C1 native Transit promotion gap after the v42
risk-banded delta-floor validation.

## New Native Promotion Evidence

Artifact:
`transit_hrl/results/scheduler_native_promotion_risk_banded_delta_floor_v42_512seed_merged`

Run:

- Domain: native Transit shared-PPO episode loop.
- Stress setting: persistent stress.
- Comparison: `native_wait_aware_replan` vs `interval_only`.
- Seeds: 512 paired seeds.
- Episodes per seed: 1.
- Profile: `risk_banded_delta_floor_v42`.

Main paired CIs:

| metric | direction | delta mean | 95% CI | status |
|---|---:|---:|---:|---|
| episode reward | increase | +0.257625 | [+0.001400, +0.654322] | supported |
| average wait min | decrease | -0.0000508 | [-0.0001113, -0.00000586] | supported |
| score | increase | +0.0000508 | [+0.00000586, +0.0001113] | supported |

Guard diagnostics:

- Wait-aware replan count increases with support: mean +0.0097656,
  CI [+0.001953, +0.019531].
- Target-headway floor rejects increase with support: mean +43.84375,
  CI [+41.6893, +45.9688].
- Final-delta floor rejects are rare: mean +0.001953, inconclusive.

Interpretation:

The previous C1 gap was that native Transit promotion could support wait/score
or reward noninferiority, but not strict reward and strict wait improvement in
the same run. The v42 profile closes that specific gap: the same 512-seed native
run supports reward increase, wait decrease, and score increase.

The key design change is conservative reward-risk targeting:

- hard pressure band instead of soft pressure cap,
- target headway band `[341s, 351s]`,
- bounded wait credit,
- reward-floor candidate scoring,
- rejection of tiny final timetable deltas below `0.03s`.

## Updated Claim Matrix

| id | status | current evidence | remaining gap |
|---|---|---|---|
| C1 native promotion | supported | v42 supports reward, wait, and score in the same 512-seed native run. | Replicate on additional stress regimes and include in final paper table. |
| C2 real demand | supported/partial | Real AFC/APC alighting-safe v2 supports score/reward and no-harm wait/alighting. | Strict wait and alighting improvement remain weak. |
| C3 order book | partial | L2 matching and synthetic/CSV-capable L3 FIFO replay paths exist. | Need larger real venue L2/L3 feeds and exchange-quality queue-priority replay. |
| C4 advanced encoder | supported | Evidence spans order-book L2, synthetic trading, real-demand Transit, and synthetic Transit. | Public-market multi-window and L3 stability remain weaker. |
| C5 leakage no-tradeoff | partial | Transit surrogate supports no-tradeoff; native paths exist. | Need native real-demand or trading native CI-supported no-tradeoff. |
| C6 theory | supported | Appendix scaffold contains theorem/proof rows for causality, leakage, promotion, wait credit, CIs, and primal-dual updates. | Final notation polish and assumption calibration. |

## Practical Next Priorities

1. Replicate v42 in at least one additional persistent-stress variant or
   pre-registered native Transit stress subset.
2. Close C5 by making native leakage no-tradeoff CI-supported, not only
   surrogate-supported.
3. Upgrade C3 with real L2/L3 order-book replay inputs and queue-priority replay
   evidence.
4. Keep C2 honest: real-demand reward is supported, but strict wait/alighting
   improvement still needs stronger native control evidence.
