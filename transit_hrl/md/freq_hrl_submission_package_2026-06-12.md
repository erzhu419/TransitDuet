# Freq-HRL Conservative Submission Package

Date: 2026-06-12

## One-Sentence Argument

Frequency-responsibility routing improves hierarchical reinforcement learning for non-stationary time-series control under the registered paired validation boundaries.

## Manuscript Thesis

The manuscript should keep one argumentative spine: frequency decomposition is a responsibility-routing principle for HRL. Low-frequency evidence belongs to upper planning, high-frequency residuals belong to lower control, persistent residuals become promotion-triggered replanning, and leakage diagnostics prevent responsibility drift.

## Title Options

1. Frequency-Separated Hierarchical Reinforcement Learning for Time-Series Control
2. Freq-HRL: Responsibility-Separated Control for Multi-Scale Time-Series Environments
3. Frequency-Routed Planning, Promotion, and Control in Hierarchical Reinforcement Learning

## Draft Abstract

Time-series control problems often couple slowly varying operating regimes with high-frequency disturbances. Conventional flat policies and generic hierarchical policies can mix these responsibilities, making recovery, attribution, and stress generalization difficult to validate. We introduce Freq-HRL, a frequency-separated hierarchical reinforcement learning protocol that routes low-frequency trend and planning signals to an upper controller, high-frequency residual control to a lower controller, and persistent shocks to a promotion-driven replanning path. The protocol includes causal frequency encoders, plan-curve actions, frequency-attributed credit, and leakage constraints that explicitly penalize responsibility drift. Across the current validation matrix, all nine conservative evidence claims are supported, including native learned promotion, native Transit real-demand service response, leakage no-tradeoff gates, baseline ablations, stress-regime coverage, public external Transit truth-source coverage, and venue-grade L2/L3 order-book replay paths. Public Transit evidence combines AFC/APC demand-driven native simulation with MBTA bus board/alight/load coverage and MTA estimated OD coverage, while order-book evidence uses LOBSTER/NASDAQ TotalView-ITCH symbol-session replay. These results support Freq-HRL as a domain-general protocol for frequency-routed time-series control, not as a completed deployment validation for every real transit agency or exchange venue.

## Core Contributions

1. A domain-general Freq-HRL protocol that separates low-frequency planning, high-frequency control, promotion-based replanning, and leakage accounting.
2. Native Transit validation with learned promotion, wait-credit, real-demand service response, and same-domain leakage no-tradeoff gates.
3. Public external Transit data coverage for MBTA board/alight/load and MTA estimated OD, kept separate from native-control performance claims.
4. Quant and order-book validation that includes baseline/ablation matrices, stress regimes, encoder variants, and venue-grade L2/L3 replay paths.
5. A formal appendix with causal encoder, leakage, promotion, credit, paired-CI, and stress-claim boundary propositions.

## Main Claim Table

| id | status | conservative_wording | boundary |
| --- | --- | --- | --- |
| C1 | supported | Native learned promotion improves reward and waiting-time metrics in the current registered stress evidence. | Best native run can support the local claim; cross-stress reward/wait improvement is evaluated separately in C7. |
| C2 | supported | Native real-demand Transit validation is supported under public AFC/APC profiles, with separate public source coverage for board/alight/load and estimated OD. | Closed for the current public AFC/APC native service-response validation and public external board/alight/load/estimated-OD source coverage. Remaining boundary: the MBTA/MTA truth files are not yet one joint agency OD/onboard-load control loop, and GTFS-ride-native replication remains optional. |
| C3 | supported | Venue-grade L2/L3 order-book replay is supported as a reproducible replay path on three LOBSTER symbol-session pairs. | Closed for the current LOBSTER/NASDAQ TotalView-ITCH venue-grade L2/L3 smoke path; remaining work is larger multi-symbol, multi-session venue replay for final paper scale. |
| C4 | supported | Advanced encoders have cross-domain support, with public-market and L3 rows kept as bounded or mixed evidence where appropriate. | Public market needs paired multi-window CIs; L3 remains mixed. |
| C5 | supported | Leakage no-tradeoff is supported only where same-domain drift reduction and performance noninferiority or strict CI gates both pass. | Closed for the current native real-demand service-response and transit surrogate leakage matrix; remaining work is independent real-agency and market-data replication. |
| C6 | supported | The formal appendix gives sufficient-condition bounds and reporting propositions rather than a universal convergence theorem. | Theory appendix now has structured theorem/proof rows; remaining work is manuscript notation polish and reviewer-facing assumption calibration. |
| C7 | supported | Promotion reward/wait improvement replicates across the current registered persistent-stress and OD-shift matrices. | Closed for the current pre-registered persistent-stress and OD-shift promotion matrices; remaining work is broader external stress replication. |
| C8 | supported | Frequency-responsibility evidence is supported against non-frequency, misrouted-frequency, no-promotion, and no-leakage alternatives. | Closed for the current baseline/ablation matrix; remaining work is adding native flat PPO/SAC/TD3 baselines for broader reviewer comparisons. |
| C9 | supported | Stress-generalization support is limited to the registered stress regimes that pass paired evidence gates. | Any missing or not-supported regime must stay outside the global stress-generalization claim. |

## Manuscript Boundary Table

| item | status | allowed_wording | disallowed_wording | evidence_hook |
| --- | --- | --- | --- | --- |
| central_claim | supported | Frequency-responsibility routing improves hierarchical reinforcement learning for non-stationary time-series control under the registered paired validation boundaries. | Freq-HRL is a universally optimal controller for every time-series deployment. | C1-C9 conservative claim matrix plus baseline, Transit, leakage, stress, encoder, and replay artifacts. |
| strong_learned_baselines | registered_missing | Flat PPO/SAC/TD3 and generic HRL are registered reviewer baselines. | Flat PPO/SAC/TD3 are complete supported baselines unless paired rows are present. | [{'baseline': 'flat_ppo', 'purpose': 'strong flat on-policy learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'flat_sac', 'purpose': 'strong off-policy entropy-regularized learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'flat_td3', 'purpose': 'strong deterministic actor-critic learned policy baseline', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}, {'baseline': 'generic_hrl_ppo', 'purpose': 'non-frequency learned HRL baseline with comparable hierarchy capacity', 'registration_status': 'registered', 'evidence_status': 'registered_missing', 'required_metrics': 'sharpe,total_return,FocusScore', 'supported_metrics': '', 'metric_statuses': '{"FocusScore": "missing", "sharpe": "missing", "total_return": "missing"}', 'paper_role': 'must_complete_or_limit', 'claim_boundary': 'This row is not credited as a strong learned baseline unless paired evidence exists for all main metrics.'}] |
| same_agency_native_transit_control | external_truth_not_control_linked | Public Transit evidence combines native public-demand service response with separate external truth-source coverage. | The current package proves one same-agency OD/onboard-load native deployment loop. | scope=real_afc_apc_external_board_alight_load_od_plus_native_service_response; field_complete=partial_external_truth_source_union |
| venue_grade_order_book_scale | venue_grade_ready | Venue-grade L2/L3 replay infrastructure is validated on the registered symbol-session pairs. | Production exchange execution or exhaustive multi-day L2/L3 replay is solved. | pairs=3 |
| formal_theory_scope | supported | The appendix gives sufficient-condition and reporting-boundary results. | The paper proves universal nonconvex actor-critic convergence. | theorems_or_propositions=9 |

## Main Baseline And Data Facts

- baseline/ablation claim status: `supported`
- scenario Freq-HRL-family win rate: `1.0`
- required positive baselines: `['allfreq_alllayers', 'hrl_raw', 'no_leakage', 'no_promotion', 'swapped', 'vanilla_rl']`
- strong learned baseline status: `registered_missing`
- real-demand evidence scope: `real_afc_apc_external_board_alight_load_od_plus_native_service_response`
- field-complete / same-agency native control: `partial_external_truth_source_union` / `external_truth_not_control_linked`
- agency supported / external-missing boundaries: `7` / `3`
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
