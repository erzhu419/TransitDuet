# Freq-HRL Top-Journal Execution Checklist

Date: 2026-06-07

This file tracks the remaining execution work after the v21 native wait-aware
promotion run closed the native reward CI but not the wait CI.

## Current Evidence

- Native wait-aware promotion v21, 8192 paired seeds:
  - reward delta: `+1.6594`, 95% CI `[+0.3398, +3.0747]`, supported.
  - wait delta: `-0.000069` minutes, 95% CI `[-0.000814, +0.000640]`,
    inconclusive.
  - reward and wait noninferiority: supported.
- Main blocker: the reward claim is now credible, but top-journal wait,
  real-demand, real-market, leakage, encoder, theory, and unified-matrix claims
  still need stronger evidence.

## 1-7 Execution Items

1. Native wait improvement CI
   - Goal: make native wait-aware promotion wait CI strictly below zero while
     preserving supported reward.
   - Current status: reward supported; wait inconclusive.
   - Local smoke update: `pressure_guarded_wait_v24` reached supported wait
     and score at 64 seeds while keeping reward positive but inconclusive:
     reward `+11.2776`, wait `-0.0016875` minutes, score `+0.00225`.
     The tighter v22/v23 gap-only profiles were too restrictive and became
     no-op in the 64-seed smoke.
   - Next action: run `pressure_guarded_wait_v24` as a scheduler multi-seed
     validation without binding shards to fixed nodes.

2. Real Transit demand wait/alighting
   - Goal: support native real-demand wait improvement and alighting throughput,
     not only score/reward or noninferiority.
   - Current status: real AFC/APC profiles drive native passengers; wait is
     positive-mixed and alighting improvement is not supported.
   - Next action: improve native real-demand control objective and rerun paired
     real-demand validation.

3. Real L2/L3 order-book validation
   - Goal: move beyond synthetic/fixture replay toward large real or realistic
     L2/L3 event replay with queue-priority sensitivity.
   - Current status: L2 matching, L3 replay, and CSV paths exist; large
     venue-grade data evidence is open.
   - Next action: add a large replay harness/manifest path and validation matrix
     that can consume multi-file L2/L3 data without changing core trading code.

4. Advanced encoder cross-domain evidence
   - Goal: show an advanced encoder consistently improves Quant and Transit
     evidence, not just exposes an interface.
   - Current status: adaptive wavelet, state-space, neural, and PINN-style paths
     exist; cross-domain CI is not closed.
   - Next action: add a unified encoder matrix with paired deltas across
     synthetic trading, public market/order-book, and Transit demand.

5. Leakage no-tradeoff native/real-data
   - Goal: show leakage reduction without performance loss in native Transit
     and real-data settings.
   - Current status: surrogate evidence is strong; native/real-data confirmation
     is incomplete.
   - Next action: add native/real-data leakage paired checks with noninferiority
     and drift-reduction criteria.

6. Formal theory
   - Goal: write paper-ready assumptions and theorem/proof sketches for
     frequency separation, leakage bounds, promotion FP/FN tradeoff,
     hierarchical wait credit, and weak constrained convergence.
   - Current status: diagnostics scaffolds exist, but formal proof text is not
     ready.
   - Next action: add a theory appendix md with definitions, propositions,
     assumptions, and proof sketches tied to measurable diagnostics.

7. Unified large matrix
   - Goal: run one claim matrix that ties together wait-aware promotion,
     real-demand Transit, encoder, leakage, and order-book evidence.
   - Current status: strong pieces exist separately; unified matrix is missing.
   - Next action: add a top-journal matrix runner/manifest and scheduler launch
     plan, then run shards opportunistically without `require_node`.

## Scheduler Rule

For retries and large seed sweeps, do not bind shards to fixed nodes unless a
node-local dependency requires it. Submit with resource hints only, for example:
`cpu_cores=32`, `ram_mb=8192`, `vram_mb=0`, `reroute_on_node_down=true`.
