# Freq-HRL Shared-Core Audit

Date: 2026-06-27

Audit question: do Transit and Quant evidence paths instantiate one Freq-HRL core, or two unrelated implementations? The answer should be reviewed through explicit adapter boundaries.

Machine-checkable audits: `transit_hrl/results/carrier_upgrade_package_latest/spec_validation.json` validates shared-core artifact paths; `transit_hrl/results/carrier_upgrade_package_latest/shared_core_validation.json` checks that core/encoder/RL modules do not import domain code and that Quant/Transit adapters use the shared training entries.

| audit_item | status | path | role | next_upgrade |
| --- | --- | --- | --- | --- |
| shared data contracts | supported | transit_hrl/freq_hrl/core/types.py | ExogenousBin and FrequencyFeatures keep domain adapters outside the core. | Keep interface frozen; domain code may only enter through adapters. |
| causal encoder interface | supported | transit_hrl/freq_hrl/encoders/base.py | Encoders expose causal low/mid/high frequency summaries. | Keep interface frozen; domain code may only enter through adapters. |
| promotion gate | supported | transit_hrl/freq_hrl/core/promotion_gate.py | Persistent high-frequency energy can trigger high-level replanning. | Keep interface frozen; domain code may only enter through adapters. |
| leakage accounting | supported | transit_hrl/freq_hrl/core/leakage.py | Upper HF power and lower LF drift are measured as responsibility leakage. | Keep interface frozen; domain code may only enter through adapters. |
| dual actor-critic core | supported | transit_hrl/freq_hrl/rl/training.py | Dual-level PPO training loop is domain-agnostic through rollout adapters. | Keep interface frozen; domain code may only enter through adapters. |
| Transit native adapter | supported | transit_hrl/freq_transitduet/runner_v3.py | Transit runner consumes Freq-HRL configs and native wait/promotion metrics. | Keep interface frozen; domain code may only enter through adapters. |
| Transit native full config | supported | transit_hrl/freq_transitduet/configs_freqduet/T_freqhrl_native_full.yaml | Native Transit instantiation of the protocol. | Keep interface frozen; domain code may only enter through adapters. |
| Trading policy adapter | supported | transit_hrl/freq_hrl/policies/ac_trading.py | Quant/trading policy path uses the same frequency-responsibility protocol. | Keep interface frozen; domain code may only enter through adapters. |
| Order-book replay adapter | supported | transit_hrl/freq_hrl/experiments/top_journal_unified_matrix.py | Order-book evidence is pulled into the same claim matrix. | Keep interface frozen; domain code may only enter through adapters. |

## Source Boundary Audit

- status: `supported`
- checked core files: `24`
- boundary violations: `0`

| adapter | status | required_symbol | role | evidence |
| --- | --- | --- | --- | --- |
| trading_ppo | supported | train_dual_ppo | Quant/trading rollout adapter calls the shared dual-level PPO loop. | `train_dual_ppo` is present in adapter source |
| transit_surrogate_ppo | supported | train_dual_ppo | Transit surrogate rollout adapter calls the same shared dual-level PPO loop. | `train_dual_ppo` is present in adapter source |
| transit_native_replay_update | supported | apply_replay_updates | Native Transit episode loop delegates PPO replay updates to the shared RL kernel. | `apply_replay_updates` is present in adapter source |
| transit_native_actor_core | supported | DualActorCriticPPO | Native Transit bridge instantiates the shared upper/lower actor-critic. | `DualActorCriticPPO` is present in adapter source |

## Reviewer-Facing Boundary

The shared core claim is supported at the training-kernel level: domain code owns rollout construction, while learning goes through `DualActorCriticPPO`, `train_dual_ppo`, or `apply_replay_updates`. A stronger final-paper claim may still report native Transit as an existing-simulator episode-loop adapter rather than pretending it is byte-identical to the synthetic rollout loop.
