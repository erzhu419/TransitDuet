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

- status: `partial`
- checked core files: `25`
- boundary violations: `0`

| adapter | status | required_symbol | role | evidence |
| --- | --- | --- | --- | --- |
| trading_ppo | supported | train_frequency_separated_ppo | Trading Freq-HRL calls the asynchronous SMDP training loop. | `train_frequency_separated_ppo` is imported and called |
| transit_surrogate_ppo | supported | train_frequency_separated_ppo | Transit surrogate must migrate to the asynchronous SMDP loop. | `train_frequency_separated_ppo` is imported and called |
| transit_native_replay_update | failed | apply_smdp_updates | Native Transit must update separate upper and lower SMDP trajectories. | `apply_smdp_updates` is not both imported and called |
| transit_native_actor_core | failed | FrequencySeparatedActorCriticPPO | Native Transit bridge must instantiate the v2 frequency-separated actor-critic. | `FrequencySeparatedActorCriticPPO` is not both imported and called |

## Reviewer-Facing Boundary

The v2 shared-core migration is incomplete: trading uses the asynchronous SMDP kernel, while Transit surrogate/native adapters still use legacy joint-PPO entries. This is a paper blocker, not a supported shared-core claim.
