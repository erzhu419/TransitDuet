# Freq-HRL Top-Journal Gap Plan

## Current Status

Freq-HRL is not fully complete at top-journal evidence level. The mechanism layer
is mostly present, and several claims have CI-supported diagnostics, but the
strongest paper claims still need deeper native, real-data, and theoretical
validation.

## Progress Added In This Pass

- Added real AFC/APC demand control replay through the shared Transit PPO
  surrogate loop.
- Added order-book spread/depth/latency stress validation.
- Added shard merge support for expanded native promotion validation.
- Added formal theory diagnostics for wait-credit residual and paired-CI width.
- Refreshed paper diagnostics to include 59 statistical checks.

New supported checks:

- `transit_real_demand_control_objective_vs_base`: supported, 6 pairs,
  delta `+1.8114`, CI `[+1.2826, +2.4539]`
- `transit_real_demand_control_wait_vs_base`: supported, 6 pairs,
  delta `-1.6741`, CI `[-2.2818, -1.1728]`
- C8 status: `supported native+real-demand`
- C10 status: `supported afc+apc-calibrated+control`

Order-book stress is improved but not a final top-journal data claim:

- `order_book_depth_adaptive_wavelet_vs_ema_sharpe`: positive-mixed, 25 pairs,
  delta `+0.2807`, CI `[-0.3942, +0.9004]`

## Hardest Remaining Gaps

### 1. Native learned promotion reward and wait proof

The learned gate fires in native Transit and gate-triggered replans are
CI-supported, but native episode reward and wait are still inconclusive:

- native learned reward delta: `+3.69`
- CI: `[-12.33, +17.85]`
- status: inconclusive because the interval crosses zero

The next target is to run more native seeds, merge the scheduler shards, and make
promotion reward/wait either supported or clearly bounded as a conditional claim.
This pass added a deterministic shard-merge utility, but the queued scheduler
shards still need completed remote outputs.

### 2. Native Transit multi-seed performance

The native shared-PPO episode loop exists, but broader native performance
validation is not complete. The current queued shards are:

- `t5822` on `jtl110cpu`
- `t5823` on `jtl110cpu2`

They must finish, sync results, and be merged into the native promotion claim
matrix before the native evidence can be considered strong.

### 3. Real Transit demand control validation

Real AFC station-hour demand and APC route-boarding now drive a shared-PPO
Transit control replay. The stronger native Transit claim still needs:

- onboard load when available
- alighting when available
- OD flow when available
- real AFC/APC demand inside the native control loop, not only surrogate replay

### 4. Real market and order-book depth

The current public-data path includes daily bars, 5-minute intraday data, an
order-book adapter fixture, and a multi-seed spread/depth/latency stress matrix.
A stronger paper still needs:

- more assets
- more markets and regimes
- larger intraday windows
- real or realistic L2/L3 order-book samples
- execution simulator sensitivity for transaction cost, slippage, and latency

### 5. Advanced encoder evidence

Adaptive wavelet, neural state-space, PINN-constrained, and state-space encoder
paths exist, but they are not yet proven to be consistently better across
domains. The next step is a cross-domain, cross-seed encoder matrix rather than
another interface-only check.

### 6. Leakage no-tradeoff beyond surrogates

Leakage/no-tradeoff diagnostics are strong in surrogate Trading and Transit, but
native Transit and real-data confirmation remain open.

### 7. Formal theory

The paper diagnostics now include theorem/report artifacts for causality,
leakage shaping, primal-dual direction, promotion tradeoff, hierarchical
wait-credit residual, and paired-CI width. A top-journal version still needs
paper-ready proofs for:

- frequency-separation assumptions
- leakage bound
- promotion false-positive / false-negative tradeoff
- primal-dual constraint/convergence argument, at least in weak form
- conditions under which the empirical frequency split is identifiable

## Execution Plan

1. Finish and merge native learned-promotion multi-seed validation.
2. Merge scheduler outputs into `transit_native_promotion_replan_expanded`.
3. Move real AFC/APC demand from surrogate replay into native Transit control.
4. Expand order-book validation from deterministic stress fixtures to larger
   real L2/L3 feeds.
5. Add paper-ready convergence/identifiability proof conditions.
6. Re-run diagnostics and push each evidence-improving step.
