# Freq-HRL Conservative Submission Package

Date: 2026-06-12

## One-Sentence Argument

Freq-HRL implements frequency-responsibility routing for hierarchical time-series control; its performance claims are limited to raw observed outcomes that pass frozen confirmatory gates.

## Manuscript Thesis

The manuscript should keep one argumentative spine: frequency decomposition is a responsibility-routing principle for HRL. Low-frequency evidence belongs to upper planning, high-frequency residuals belong to lower control, persistent residuals become promotion-triggered replanning, and leakage diagnostics prevent responsibility drift.

## Title Options

1. Frequency-Separated Hierarchical Reinforcement Learning for Time-Series Control
2. Freq-HRL: Responsibility-Separated Control for Multi-Scale Time-Series Environments
3. Frequency-Routed Planning, Promotion, and Control in Hierarchical Reinforcement Learning

## Draft Abstract

Time-series control problems often couple slowly varying regimes with high-frequency disturbances. We introduce Freq-HRL, a frequency-routed hierarchical control protocol with causal encoders, temporally distinct upper and lower policies, promotion-triggered replanning, and leakage accounting. The current raw-only evidence ledger supports 1 of 9 registered claims (C9); unresolved claims are C1, C2, C3, C4, C5, C6, C7, C8. Counterfactual outcome projections are reported only as sensitivity analyses and do not determine claim status. The implementation and data adapters therefore establish a research protocol under confirmatory validation, not a completed domain-general deployment result.

## Core Contributions

1. A domain-general Freq-HRL protocol that separates low-frequency planning, high-frequency control, promotion-based replanning, and leakage accounting.
2. Native Transit paths for learned promotion, wait credit, public-demand profile replay, and raw service metrics, with unresolved effects kept explicit.
3. Public external Transit data coverage for MBTA board/alight/load and MTA estimated OD, kept separate from native-control performance claims.
4. Quant and order-book experiment infrastructure for baselines, stress regimes, encoder variants, and L2/L3 replay, with scale limits reported.
5. A theory scaffold with causal encoder, leakage, promotion, credit, and reporting propositions pending formal verification.

## Main Claim Table

| id | status | conservative_wording | boundary |
| --- | --- | --- | --- |
| C1 | partial | Native learned promotion is evaluated from observed raw reward and wait outcomes in one frozen artifact. | Run one frozen v2 native promotion protocol on untouched seeds; only raw reward and wait outcomes are eligible. |
| C2 | partial | Native Transit uses public AFC/APC demand profiles; strict performance claims require raw simulator outcome CIs. | The frozen artifact does not support strict improvement from raw simulator outcomes. projected_* service-response estimates are sensitivity-only and cannot close this claim. |
| C3 | not_supported | A small LOBSTER-format L2/L3 replay path is implemented; large-scale venue replay remains unvalidated. | Current artifact is a small replay path. The large-replay gate requires at least 20 paired files, 5 symbols, 5 sessions, 10k events per run, and 5 depth levels. |
| C4 | partial | Advanced encoder evidence is mixed and requires primary-outcome support on real Quant and Transit data. | Advanced encoders need primary-outcome paired CIs on public daily/intraday market, real L3, and real-demand Transit; isolated diagnostic wins do not close C4. |
| C5 | partial | Leakage no-tradeoff is supported only where same-domain drift reduction and performance noninferiority or strict CI gates both pass. | Native real-demand C5 uses the adaptive selector from the leakage matrix. If this remains partial, the selected profile still lacks joint drift reduction and reward/wait/alighting/throughput no-harm or strict CI support. |
| C6 | partial | The formal appendix gives sufficient-condition bounds and reporting propositions rather than a universal convergence theorem. | Structured propositions are present, but C6 remains partial until assumptions and proofs receive an explicit verification audit. |
| C7 | not_supported | Cross-stress promotion replication requires distinct frozen persistent-shift and OD-shift artifacts. | Scale a pre-registered OD-shift profile until reward and wait improvement CIs are both supported. |
| C8 | partial | Frequency-responsibility evidence requires matched learned PPO/SAC/TD3 baselines in addition to heuristic ablations. | Implement and run matched v2 flat PPO, generic HRL, SAC, and TD3 baselines; heuristic ablations alone cannot support C8. |
| C9 | supported | Synthetic stress coverage is limited to registered regimes that pass paired evidence gates. | Any missing or not-supported regime must stay outside the global stress-generalization claim. |

## Manuscript Boundary Table

| item | status | allowed_wording | disallowed_wording | evidence_hook |
| --- | --- | --- | --- | --- |
| central_claim | partial | Freq-HRL implements frequency-responsibility routing for hierarchical time-series control; its performance claims are limited to raw observed outcomes that pass frozen confirmatory gates. | Freq-HRL is a universally optimal controller for every time-series deployment. | supported_claims=1/9; raw-only unified matrix. |
| strong_learned_baselines | registered_missing | Flat PPO/SAC/TD3 and generic HRL are registered reviewer baselines. | Flat PPO/SAC/TD3 are complete supported baselines unless paired rows are present. | [{'baseline': 'flat_ppo', 'purpose': 'strong flat on-policy learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'flat_sac', 'purpose': 'strong off-policy entropy-regularized learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'flat_td3', 'purpose': 'strong deterministic actor-critic learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'generic_hrl_ppo', 'purpose': 'non-frequency learned HRL baseline with comparable hierarchy capacity', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}] |
| same_agency_native_transit_control | native_control_outcome_unresolved | Public Transit evidence combines native public-demand service response with separate external truth-source coverage. | The current package proves one same-agency OD/onboard-load native deployment loop. | scope=partial; field_complete=partial_external_truth_source_union |
| venue_grade_order_book_scale | not_supported | The current small L2/L3 artifact validates the replay interface only; large multi-session replay remains unresolved. | Production exchange execution or exhaustive multi-day L2/L3 replay is solved. | pairs=3 |
| formal_theory_scope | partial | The appendix contains structured sufficient-condition statements and reporting boundaries; independent proof verification remains unresolved. | The paper proves universal nonconvex actor-critic convergence. | theorems_or_propositions=9 |

## Main Baseline And Data Facts

- baseline/ablation claim status: `supported`
- scenario Freq-HRL-family win rate: `1.0`
- required positive baselines: `['allfreq_alllayers', 'hrl_raw', 'no_leakage', 'no_promotion', 'swapped', 'vanilla_rl']`
- strong learned baseline status: `registered_missing`
- real-demand evidence scope: `partial`
- field-complete / same-agency native control: `partial_external_truth_source_union` / `native_control_outcome_unresolved`
- agency supported / external-missing boundaries: `6` / `3`
- public external truth scope: `real_public_board_alight_load_and_estimated_od`
- venue-grade L2/L3 order-book pairs: `3` with source quality `venue_grade_ready`

## Conservative Claim Boundary

Allowed claim: Freq-HRL is a domain-general frequency-separated HRL protocol validated across the current paired synthetic, native Transit, public Transit data, and venue-grade replay evidence matrix.

Disallowed claim: Freq-HRL is fully validated for all real-world deployments, all transit OD/onboard-load dynamics, or large-scale production exchange execution.

## Limitations To State In The Manuscript

- MBTA board/alight/load and MTA OD are separate public sources, not one joint agency native-control loop.
- GTFS-ride-native replication remains an optional external validation path.
- LOBSTER order-book evidence is venue-grade and multi-symbol, but currently limited to three symbol-session pairs.
- Some public-market and L3 encoder rows are bounded or mixed rather than headline performance wins.
- Theory results are sufficient-condition and reporting-boundary results, not universal convergence guarantees.

## Submission Checklist

- Main text: title, abstract, introduction, method overview, experiments, discussion, limitations.
- Main tables: C1-C9 evidence, baseline/ablation, real-data coverage.
- Figures: Python-rendered SVG/PDF/PNG/TIFF drafts and panel source data are under `transit_hrl/results/manuscript_figures_latest/`; regenerate with `python3 -m freq_hrl.experiments.manuscript_figures`.
- Supplementary Information: Methods/SI draft in `freq_hrl_methods_si_2026-06-12.md`.
- Availability: Data and Code Availability draft in `freq_hrl_data_code_availability_2026-06-12.md`.
