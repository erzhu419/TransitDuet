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
- Scheduler update after items 1-7:
  - Native promotion v24, 2048 paired seeds: wait delta `-0.0002676` minutes,
    95% CI `[-0.0004385, -0.0001225]`, supported; reward delta `+0.2494`,
    CI crosses zero, reward noninferiority supported.
  - Native real-demand v2, 24 paired AFC/APC seeds: score and reward supported;
    wait is positive-mixed; alighting throughput is not supported.
  - Order-book manifest smoke: manifest consumed both L2 and L3 files and wrote
    paired checks; this validates the large-replay path, not a large real feed.
  - Encoder matrix: supported evidence appears in order-book L2, synthetic
    trading, Transit real-demand estimator, and Transit synthetic demand.
  - Leakage matrix: Transit surrogate no-tradeoff is supported; trading and
    native real-demand remain partial.
  - Unified top-journal matrix v2: 6 claims total, 3 supported, 3 partial,
    0 missing.

## 2026-06-08 Reward/Throughput/Leakage Attack

- Native promotion now has a `reward_floor_throughput_v25` persistent-stress
  profile.  It starts from the wait-supported v24 policy and adds:
  - a causal reward-floor score before accepting promotion replans;
  - a throughput proxy guard built from upper-state fleet/gap/holding and
    high-frequency wait feedback;
  - an adaptive drift penalty that scales promotion shifts down when HF energy
    dominates the LF promotion signal;
  - a target-headway floor to avoid over-aggressive timetable compression.
- Native real-demand treatment reuses the same reward-floor, throughput, and
  adaptive-drift guards, so AFC/APC validation can test wait/alighting and
  leakage tradeoffs in the same path.
- New diagnostics are emitted into native rows and paired checks:
  `shared_ppo_reward_floor_guard_rejects`,
  `shared_ppo_throughput_guard_rejects`,
  `shared_ppo_target_headway_floor_rejects`,
  `shared_ppo_wait_replan_adaptive_drift_scale_mean`,
  `shared_ppo_wait_replan_throughput_score_mean`, and
  `shared_ppo_wait_replan_reward_floor_score_mean`.
- Scheduler submissions:
  - v25 native promotion 512-seed shards: `t7715`, `t7716`, `t7717`,
    `t7719`, `t7720`, `t7721`, `t7722`, `t7723`.
  - v25 promotion merge: `t7734`, waiting for the 8 shard summaries.
  - real-demand guarded v3 AFC/APC 24-pair validation: `t7726`.
- Unified top-journal and leakage matrices now prefer v25/v3 artifacts and
  fall back to v24/v2 when those results are not present.

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
   - Scheduler result: 2048 paired seeds support wait improvement and reward
     noninferiority, but reward improvement itself remains inconclusive.

2. Real Transit demand wait/alighting
   - Goal: support native real-demand wait improvement and alighting throughput,
     not only score/reward or noninferiority.
   - Current status: real AFC/APC profiles drive native passengers; wait is
     positive-mixed and alighting improvement is not supported.
   - Implementation update: `native_real_freqhrl` now uses the same guarded
     wait-aware learned replan policy as the best native-promotion candidate:
     promotion no longer only flips the generic timetable replanner; it passes
     pressure, wait, hold, gap, target-headway projection, terminal early-cap,
     and actor-base trust controls into the shared PPO native loop.
   - Merge update: added a real-demand shard merger so scheduler shards can be
     combined into one paired AFC/APC CI report without local post-processing.
   - Next action: submit paired AFC/APC real-demand validation through scheduler,
     not local execution, and check wait/alighting CI plus the replan diagnostics.
   - Scheduler result: score/reward supported; wait positive-mixed; alighting
     not supported. Native drift metrics are now present in the drift-aware
     merged artifact.

3. Real L2/L3 order-book validation
   - Goal: move beyond synthetic/fixture replay toward large real or realistic
     L2/L3 event replay with queue-priority sensitivity.
   - Current status: L2 matching, L3 replay, and CSV paths exist; large
     venue-grade data evidence is open.
   - Implementation update: added a manifest-driven large replay runner that
     consumes multi-file L2/L3 CSV collections with venue/symbol/session
     metadata, dispatches them through the existing L2 matching and L3 FIFO
     replay engines, and writes one coverage/paired-check report.
   - Next action: point the manifest runner at a larger real or venue-realistic
     L2/L3 dataset and schedule the replay matrix.
   - Scheduler result: fixture L2/L3 manifest smoke passed and wrote coverage /
     paired-check artifacts.

4. Advanced encoder cross-domain evidence
   - Goal: show an advanced encoder consistently improves Quant and Transit
     evidence, not just exposes an interface.
   - Current status: adaptive wavelet, state-space, neural, and PINN-style paths
     exist; cross-domain CI is not closed.
   - Implementation update: added a cross-domain encoder matrix builder that
     reads existing synthetic trading, public market, L2/L3 order-book, and
     Transit demand artifacts, then emits one paired-check/domain-summary report
     with `supported`, `positive_mixed`, `not_supported`, and `summary_only`
     statuses.
   - Next action: schedule the matrix refresh after the new order-book and
     real-demand runs finish, then use it to target encoder reruns.
   - Scheduler result: matrix built 96 checks; at least one supported check
     appears in four domains.

5. Leakage no-tradeoff native/real-data
   - Goal: show leakage reduction without performance loss in native Transit
     and real-data settings.
   - Current status: surrogate evidence is strong; native/real-data confirmation
     is incomplete.
   - Implementation update: added a leakage no-tradeoff matrix that requires
     both drift/leakage reduction and performance noninferiority in the same
     domain before calling a no-tradeoff verdict. Native real-demand artifacts
     are explicitly marked as performance/no-harm evidence when drift metrics
     are absent.
   - Native metric update: real-demand validation now extracts
     `LowerLFDrift` and `UpperHFPower` from the native shared-PPO summaries so
     future native AFC/APC reruns can test leakage/no-tradeoff directly.
   - Merge update: the real-demand shard merger now rebuilds rows from compact
     payload summaries, so existing shards can expose newly added native drift
     metrics without rerunning seeds. The drift-aware merged artifact uses the
     `transit_native_real_demand_waitaware_v2_24seed_merged_drift` result dir.
   - Next action: schedule the matrix after the wait-aware real-demand merge,
     then add native LowerLFDrift metrics if the verdict remains
     `performance_noharm_only`.
   - Scheduler result: Transit real surrogate reached `no_tradeoff_supported`;
     native real-demand and trading remain `partial`.

6. Formal theory
   - Goal: write paper-ready assumptions and theorem/proof sketches for
     frequency separation, leakage bounds, promotion FP/FN tradeoff,
     hierarchical wait credit, and weak constrained convergence.
   - Current status: diagnostics scaffolds exist, but formal proof text is not
     ready.
   - Implementation update: extended the theory appendix generator with an
     explicit weak projected primal-dual constraint bound, in addition to the
     existing causal encoder, leakage-shaped return, promotion FP/delay,
     hierarchical credit residual, and paired-CI arguments.
   - Next action: regenerate the appendix artifact through scheduler after the
     latest diagnostics are refreshed.
   - Scheduler result: appendix artifact regenerated with the weak
     primal-dual bound included.

7. Unified large matrix
   - Goal: run one claim matrix that ties together wait-aware promotion,
     real-demand Transit, encoder, leakage, and order-book evidence.
   - Current status: strong pieces exist separately; unified matrix is missing.
   - Implementation update: added a unified top-journal evidence matrix runner
     that reads native promotion, native real-demand, order-book manifest,
     encoder matrix, leakage matrix, and theory appendix artifacts, then emits
     claim-level `supported` / `partial` / `missing` statuses and remaining gaps.
   - Next action: schedule the unified matrix after the v24 promotion shards and
     theory appendix refresh finish.
   - Scheduler result: final matrix v2 produced 3 supported claims and 3 partial
     claims; no claim is missing.

## Scheduler Rule

For retries and large seed sweeps, do not bind shards to fixed nodes unless a
node-local dependency requires it. Submit with resource hints only, for example:
`cpu_cores=32`, `ram_mb=8192`, `vram_mb=0`, `reroute_on_node_down=true`.
