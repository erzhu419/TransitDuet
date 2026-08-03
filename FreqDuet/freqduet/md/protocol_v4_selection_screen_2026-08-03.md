# FreqDuet Protocol v4 Selection Screen

Date: 2026-08-03

## Purpose

This screen decides whether the physically executable and deployable v4 rebuild
retains the frequency-separated method claim. It is model selection, not final
confirmation evidence. No v3 result is eligible for reuse or relabeling.

## Locked execution contract

- Git tag: `freqduet-v4-selection-v1`
- Run name: `protocol_v4_selection_ep40_s20_e8_v1`
- Compute: scheduler-tracked CPU jobs on `node001-node006`
- Environment: isolated `freqduet-cpu-py310`; no default environment changes
- Training: 40 episodes, checkpoint episode 39, no resume
- Sharding: one config/seed training job per scheduler task
- Evaluation: frozen deterministic policies on identical scenario tapes
- Heavy intermediate checkpoints are removed after frozen evaluation

Configurations, with `main` as the paired reference:

1. `F_freqduet_protocol_v4_main_hiro`
2. `F_freqduet_protocol_v4_csac_hiro`
3. `F_freqduet_protocol_v4_nofreq_hiro`
4. `F_freqduet_protocol_v4_rawhistory_hiro`
5. `F_freqduet_protocol_v4_allfreq_hiro`
6. `F_freqduet_protocol_v4_nopromotion_hiro`
7. `F_freqduet_protocol_v4_noleakage_hiro`
8. `F_freqduet_protocol_v4_nodriftfb_hiro`
9. `F_freqduet_protocol_v4_noprior_hiro`
10. `F_freqduet_protocol_v4_continuous_holding_hiro`
11. `F_freqduet_protocol_v4_nolowercontext_hiro`

Training seeds:

`211,227,239,251,263,277,293,307,317,331,347,359,373,389,401,419,431,443,457,467`

Frozen evaluation seeds:

`31013,31019,31033,31039,31051,31063,31069,31079`

These seed sets are disjoint from protocol-v3 selection. Confirmation seeds
remain untouched until this screen selects one locked method.

## Decision rule

The primary endpoint is `service_cost_restricted`, evaluated with paired
hierarchical bootstrap over training seeds and frozen scenario tapes. Generic
`service_cost` is retained only as a consistency view because every v4 row uses
the restricted-wait objective. Pairwise inference reports confidence intervals,
effect sizes, raw p-values, and Holm-adjusted p-values.

A candidate can replace `main` only if it improves the primary endpoint without
a material regression in unserved-passenger rate, trip launch/completion rate,
headway CV, or fleet overshoot. All physical, observation, source-hash,
resolved-config-hash, and scenario-tape invariants must pass first.

Interpretation is fixed before results:

- If `nofreq` or `rawhistory` is best, the frequency-separated efficacy claim
  has failed and the method must be redesigned rather than renamed post hoc.
- If a mechanism ablation is better, remove or redesign that mechanism and use
  fresh confirmation seeds.
- If standard constrained SAC matches or beats the ensemble, prefer the simpler
  optimizer unless a later untouched confirmation reverses the result.
- If no variant materially improves `main`, retain `main` and move to untouched
  100/200-episode confirmation and external baselines.
