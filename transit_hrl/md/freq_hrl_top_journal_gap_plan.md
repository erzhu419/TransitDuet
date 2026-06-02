# Freq-HRL Top-Journal Gap Plan

## Current Status

Freq-HRL is not fully complete at top-journal evidence level. The mechanism layer
is mostly present, and several claims have CI-supported diagnostics, but the
strongest paper claims still need deeper native, real-data, and theoretical
validation.

## Progress Added In This Pass

- Added real AFC/APC demand control replay through the shared Transit PPO
  surrogate loop.
- Added native AFC/APC-profile passenger generation inside the copied Transit
  simulator. Public AFC/APC temporal and station-intensity profiles now drive
  native passenger objects, boarding, alighting, and onboard-load metrics.
- Added order-book spread/depth/latency stress validation.
- Added an L2 market/passive-queue matching simulator with latency, partial
  fills, multi-level fills, slippage, best-level queue-priority proxy, and real
  L2 CSV input support.
- Added synthetic/CSV L3 order-event replay with add/cancel/trade events,
  order IDs, FIFO queue priority, and passive agent order fills.
- Added shard merge support for expanded native promotion validation.
- Completed and merged an expanded native learned-promotion validation batch:
  the original native batch plus two corrected 32-seed scheduler shards now
  cover 76 paired seeds / 228 native rows.
- Added formal theory diagnostics for wait-credit residual and paired-CI width.
- Refreshed paper diagnostics to include 101 statistical checks.

New supported checks:

- `transit_real_demand_control_objective_vs_base`: supported, 6 pairs,
  delta `+1.8114`, CI `[+1.2826, +2.4539]`
- `transit_real_demand_control_wait_vs_base`: supported, 6 pairs,
  delta `-1.6741`, CI `[-2.2818, -1.1728]`
- C3 status: `supported learned; native learned-gate path`
- C8 status: `supported native; real-demand wait-noninferior`
- C10 status: `supported afc+apc-calibrated+native-score`
- Native learned-gate reward improvement: positive-mixed, 76 pairs,
  delta `+5.2023`, CI `[-5.4383, +16.2658]`
- Native learned-gate reward noninferiority: supported, margin `15.0`,
  delta `+5.2023`, CI `[-5.4383, +16.2658]`
- Native learned-gate wait improvement: not supported, 76 pairs,
  delta `+0.0169`, CI `[-0.0029, +0.0472]`
- Native learned-gate wait noninferiority: inconclusive, margin `0.01`
  minutes, delta `+0.0169`, CI `[-0.0029, +0.0472]`
- Native learned-gate gate-triggered replans: supported, 76 pairs,
  delta `+1.7368`, CI `[+1.5921, +1.8684]`
- Native real-demand control score: supported, 6 pairs, delta `+99.6725`,
  CI `[+62.3044, +137.0299]`
- Native real-demand reward: supported, 6 pairs, delta `+98.7658`,
  CI `[+59.9997, +137.1515]`
- Native real-demand wait: positive-mixed, 6 pairs, delta `-0.0830`,
  CI `[-0.2248, +0.0567]`
- Native real-demand wait noninferiority: supported, margin `0.10` minutes,
  delta `-0.0833`, CI `[-0.2251, +0.0564]`
- Native real-demand alighted passengers: not supported, 6 pairs,
  delta `-4.8333`, CI `[-9.8333, -0.8333]`
- Native real-demand alighted noninferiority: supported, margin `19.7068`
  passengers, delta `-4.8333`, CI `[-9.8333, -0.8333]`

Order-book stress is improved but not a final top-journal data claim:

- `order_book_depth_adaptive_wavelet_vs_ema_sharpe`: positive-mixed, 25 pairs,
  delta `+0.2807`, CI `[-0.3942, +0.9004]`
- `order_book_matching_state_space_vs_ema_sharpe`: supported, 30 pairs,
  delta `+366.2083`, CI `[+299.1326, +432.6357]`
- `order_book_matching_adaptive_wavelet_vs_ema_sharpe`: positive-mixed, 30
  pairs, delta `+1.4779`, CI `[-1.3087, +3.8865]`
- `order_book_l3_adaptive_wavelet_vs_ema_sharpe`: positive-mixed, 10 pairs,
  delta `+21.5438`, CI `[-7.2826, +53.6276]`
- `order_book_l3_adaptive_wavelet_vs_ema_fill_rate`: positive-mixed, 10 pairs,
  delta `+0.0029`, CI `[-0.0033, +0.0096]`

Boundary of this pass:

- Native real demand uses public AFC/APC profiles mapped onto the copied native
  corridor. It is a native passenger loop, but not exact public OD geometry.
- L2 matching can consume real multi-level CSVs and now includes a best-level
  passive queue-priority proxy.
- L3 replay can consume CSV add/cancel/trade events and now simulates FIFO queue
  priority for passive agent fills. The committed validation still uses
  synthetic L3 tapes; large real venue-grade L3 feeds remain open.

## Hardest Remaining Gaps

### 1. Native learned promotion reward and wait proof

The learned gate fires in native Transit and gate-triggered replans are
CI-supported. Expanded 76-seed validation improved the reward estimate, but
did not close the strong reward/wait improvement claim:

- native learned reward delta: `+5.2023`
- reward CI: `[-5.4383, +16.2658]`
- reward status: positive-mixed because the interval still crosses zero
- native learned reward noninferiority: supported with margin `15.0`
- native learned wait delta: `+0.0169` minutes
- wait CI: `[-0.0029, +0.0472]`
- wait status: not supported; wait noninferiority is now inconclusive under the
  strict `0.01` minute margin

The next target is not more blind seeds with the same gate. A local 8-seed
gate sweep showed that `max_total_replans=1` hurts reward/wait, while stronger
threshold/low-frequency guards mostly degenerate to no-op. The remaining path
is a learned replan policy that changes the high-level timetable action under a
wait-aware objective, not only a gate that refreshes the current plan.

### 2. Native Transit multi-seed performance

The native shared-PPO episode loop exists, and expanded promotion shards now
run end-to-end. The corrected scheduler tasks were:

- `t5952` on `jtl110cpu2`: done, 32 seeds
- `t5953` on `jtl110cpu2`: done, 32 seeds

The original `jtl110cpu` route (`t5822` / `t5951`) failed before launch because
the scheduler could not upload the Windows payload to the node. The shard was
therefore rerouted as `t5953` to `jtl110cpu2`. Broader native performance is
still not complete because the expanded CI did not prove wait/reward
improvement.

### 3. Real Transit demand control validation

Real AFC station-hour demand and APC route-boarding now drive a shared-PPO
Transit control replay, and AFC/APC profiles now drive the native passenger
loop. The stronger native Transit claim still needs:

- onboard load when available
- alighting when available
- OD flow when available
- exact AFC/APC OD geometry instead of profile-to-corridor mapping
- supported wait/alighting improvement CIs in the native real-demand loop.
  Bounded noninferiority is now supported, but improvement remains open.

### 4. Real market and order-book depth

The current public-data path includes daily bars, 5-minute intraday data, an
order-book adapter fixture, a multi-seed spread/depth/latency stress matrix,
and an L2 market/passive-queue matching simulator that can read real multi-level
CSVs. It now also includes a synthetic/CSV L3 add/cancel/trade replay path.
A stronger paper still needs:

- more assets
- more markets and regimes
- larger intraday windows
- real or realistic L2/L3 order-book samples
- large real exchange L3 event feeds beyond the current synthetic/CSV replay
- venue-specific matching details such as hidden/iceberg liquidity and auction
  handling
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

1. Replace the current native learned gate with a wait-aware learned replan
   policy that can change timetable actions after promotion, then re-run the
   expanded native CI.
2. Improve native real-demand loop so wait and alighting throughput improvement
   CIs are supported; for now score/reward plus wait/alighting noninferiority
   are supported.
3. Expand order-book validation from synthetic L2/L3 fixtures to larger real
   venue-grade L2/L3 feeds.
4. Add paper-ready convergence/identifiability proof conditions.
5. Re-run diagnostics and push each evidence-improving step.
