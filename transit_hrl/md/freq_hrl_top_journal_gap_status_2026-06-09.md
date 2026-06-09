# Freq-HRL Top-Journal Gap Status, 2026-06-09

This note records the latest post-run gap state after the native promotion v31/v24,
native real-demand alighting-safe v2, encoder, leakage, order-book, and upgraded
theory-appendix artifacts were re-indexed into the unified evidence matrix.

## Current Matrix

Latest artifact:
`transit_hrl/results/top_journal_unified_matrix_latest/summary.json`

Summary:

- Supported claims: 3
- Partial claims: 3
- Missing / not supported claims: 0

Claim-level state:

| id | status | current evidence | remaining gap |
|---|---|---|---|
| C1 | partial | Best reward artifact is v21: reward supported, wait no-harm supported. Best wait artifact is v24 fixed 2048 seeds: wait supported, reward no-harm supported. | Need reward and wait both CI-supported in the same native run. |
| C2 | supported | Native real AFC/APC demand alighting-safe v2 supports score/reward and wait/alighting no-harm. | Strict wait and alighting improvement CIs are still inconclusive. |
| C3 | partial | L2 matching has 8 supported checks; synthetic/CSV-capable L3 FIFO replay has 8 positive checks. | Need larger real venue L2/L3 feeds and exchange-quality queue-priority replay. |
| C4 | supported | Encoder evidence spans order-book L2, synthetic trading, transit real demand, and transit synthetic demand. | Public-market paired multi-window CIs and L3 stability remain weak. |
| C5 | partial | Transit surrogate has CI-supported leakage no-tradeoff; native real-demand, trading constraint, and PPO primal-dual are partial/summary-only. | Need native real-demand or trading native CI-supported no-tradeoff, not just surrogate or summary-only trajectory. |
| C6 | supported | Theory appendix now contains 7 structured theorem/proof rows covering causality, leakage shaping, promotion false positives, detection delay, wait-credit residuals, paired CI width, and primal-dual leakage updates. | Remaining work is manuscript notation polish and reviewer-facing assumption calibration. |

## Practical Next Priorities

1. Close C1 by designing a native promotion profile whose wait improvement is
   strict while preserving the v21 reward support. Current v24 fixed 2048 seeds
   gives strict wait support and reward no-harm, but reward CI still crosses 0.
2. Upgrade C5 by adding native paired leakage/no-tradeoff evidence with core
   metrics only: drift reduction plus reward/wait/alighting noninferiority.
3. Upgrade C3 with real L2/L3 order-book replay inputs instead of synthetic
   fixture tapes.
4. Keep C6 manuscript-ready by aligning notation and assumptions with the final paper draft after C1/C5 evidence stabilizes.

The current gap is no longer missing implementation paths. The remaining gap is
mostly claim-strength: native promotion wait improvement, native leakage
no-tradeoff, and real order-book scale.
