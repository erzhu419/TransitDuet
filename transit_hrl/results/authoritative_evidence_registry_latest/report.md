# Freq-HRL Authoritative Evidence Ledger

Date: 2026-08-26

This is the only manuscript claim ledger. Unregistered artifacts and the old independent claim generators are excluded by default.

| Evidence | Domain | Stage | Decision | Paper use | Positive claim |
|---|---|---|---|---|---:|
| mujoco_v12_responsibility_confirmatory | mujoco_control | confirmatory | supported | positive_main_or_si | true |
| mujoco_v13_behavioral_confirmatory | mujoco_control | confirmatory | not_supported | mixed_or_negative_main_or_si | false |
| quant_v74_matched_baseline_confirmatory | quant_synthetic_control | confirmatory | mixed | mixed_or_negative_main_or_si | false |
| mujoco_v14_endpoint_aligned_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_1_crossed_upper_pd_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_2_physical_router_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_3_partial_router_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_4_router_homotopy_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_5_paired_anchor_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_6_conservative_transfer_screen | mujoco_control | development | no_behavior_safe_candidate | development_only | false |
| mujoco_v14_7_joint_learned_projection_preflight | mujoco_control | development | comparator_confounded | development_only | false |
| mujoco_v14_8_latent_matched_preflight | mujoco_control | development | no_joint_candidate | development_only | false |
| mujoco_v14_9_asymmetric_feasibility_preflight | mujoco_control | development | deployment_constraint_misaligned | development_only | false |
| mujoco_v14_10_deployment_aligned_preflight | mujoco_control | development | do_not_expand | development_only | false |
| mujoco_v14_11_iterative_projection_preflight | mujoco_control | development | do_not_expand | development_only | false |
| mujoco_v14_12_groupwise_robust_preflight | mujoco_control | development | do_not_expand | development_only | false |
| mujoco_v14_13_anchor_replay_trust_preflight | mujoco_control | development | do_not_expand | development_only | false |
| mujoco_v14_14_closed_loop_actor_guard_preflight | mujoco_control | development | do_not_expand | development_only | false |
| mujoco_v14_15_closed_loop_restoration_filter_preflight | mujoco_control | development | expand_to_multiseed_screen | development_only | false |
| mujoco_v14_15_restoration_multiseed_development_r2 | mujoco_control | development | candidate_not_ready_for_confirmation | development_only | false |
| mujoco_v14_16_crossed_restoration_mechanism_development_r5 | mujoco_control | development | primary_mechanism_not_ready | development_only | false |
| legacy_c1_c9_matrix_snapshot | cross_domain_legacy | legacy | excluded_legacy | excluded_legacy | false |
| legacy_paper_diagnostics_snapshot | cross_domain_legacy | legacy | excluded_legacy | excluded_legacy | false |

## Allowed Wording

### mujoco_v12_responsibility_confirmatory

On the frozen v12 protocol, Freq-HRL reduced the registered responsibility-space lower-LF diagnostic and met return noninferiority in HalfCheetah-v5, Hopper-v5, and Walker2d-v5. This result does not establish separation of the pre-routing lower action or the upper-action high-frequency budget.

Forbidden: MuJoCo v12 proves raw behavioral frequency separation or universal no-tradeoff.

### mujoco_v13_behavioral_confirmatory

The stricter frozen v13 behavioral claim failed: HalfCheetah-v5 missed the raw lower-LF reduction gate and Hopper-v5 missed the upper-HF budget; Walker2d-v5 passed all registered gates. Return noninferiority and responsibility-space reduction passed in all three tasks.

Forbidden: Freq-HRL has confirmatory support for raw behavioral frequency separation across all MuJoCo tasks.

### quant_v74_matched_baseline_confirmatory

Across 24 independent training replicates, eight held-out paths per replicate, and six registered scenarios, Freq-HRL v7.4 supported improvement in 8 of 12 Holm-controlled pooled contrasts. Its return was significantly worse than matched generic HRL-GRU-PPO, while its return contrast with matched SAC and both drift contrasts with generic HRL variants were inconclusive.

Forbidden: Freq-HRL uniformly dominates all matched PPO, recurrent HRL, SAC, and TD3 baselines.

### mujoco_v14_endpoint_aligned_screen

The v14 development screen found no fixed endpoint-aligned coefficient that passed all four safety gates in all three MuJoCo tasks; it motivates adaptive or constrained upper-policy training. These outcomes are not confirmatory evidence.

Forbidden: MuJoCo v14 supports a positive performance or behavior claim, or supplies held-out confirmation for a later algorithm.

### mujoco_v14_1_crossed_upper_pd_screen

The v14.1 development screen evaluated crossed-condition checkpoint selection, the strongest v14 static arm, and three upper primal-dual rates. No arm passed the preregistered return and behavioral gates in all 15 environment-by-disturbance conditions; every candidate passed zero complete condition gates. These outcomes are development evidence only.

Forbidden: MuJoCo v14.1 validates behavior-safe upper primal-dual Freq-HRL or supplies confirmatory evidence for any selected arm.

### mujoco_v14_2_physical_router_screen

The v14.2 development screen corrected the leakage-cost scale and showed that full causal high-pass routing reduced raw and responsibility lower-LF drift in every registered environment-by-disturbance condition. No arm was behavior-safe overall: the best arm passed 2 of 15 complete condition gates, full routing harmed return or lower-action activity, and upper primal-dual variants saturated in at least one environment. These outcomes motivate partial-strength routing and are not confirmatory evidence.

Forbidden: MuJoCo v14.2 validates physical behavioral separation, no-tradeoff, or any selected confirmatory Freq-HRL algorithm.

### mujoco_v14_3_partial_router_screen

The v14.3 development screen tested seven fixed partial-strength causal lower-action routers on fresh seeds. Every arm reduced responsibility-space lower-LF drift in all 15 environment-by-disturbance conditions, but no arm jointly passed return and raw-behavior gates everywhere. The best arm passed 4 of 15 complete conditions and 10 of 15 strict raw-drift gates. These outcomes motivate a training-time routing curriculum and are not confirmatory evidence.

Forbidden: MuJoCo v14.3 validates fixed partial-strength routing, physical no-tradeoff, or any selected confirmatory Freq-HRL algorithm.

### mujoco_v14_4_router_homotopy_screen

The v14.4 development screen exposed router strength to both policies and tested seven frozen constant or delayed homotopy schedules on fresh seeds. No schedule was behavior-safe across all 15 environment-by-disturbance conditions. The best joint arm passed 3 of 15 complete conditions. The fastest ramp met return noninferiority in 10 of 15 conditions but met responsibility and raw-drift gates in only 5 each, indicating policy compensation rather than stable physical separation. These outcomes motivate paired checkpoint continuation and are not confirmatory evidence.

Forbidden: MuJoCo v14.4 validates router homotopy, physical no-tradeoff, or any selected confirmatory Freq-HRL algorithm.

### mujoco_v14_5_paired_anchor_screen

The v14.5 development screen started every candidate and its compute-matched direct comparator from the same serialized policy and optimizer state. Actor-space proximal constraints stabilized latent policy movement, and two arms met both raw and responsibility drift gates in all 15 conditions. No arm was behavior-safe overall: every routed arm met return noninferiority in only the five Walker2d conditions, Hopper missed the upper-HF budget, and no arm met the trained-checkpoint gate. This identifies a function-discontinuous router intervention rather than optimizer initialization as the remaining defect.

Forbidden: MuJoCo v14.5 validates paired proximal routing, physical no-tradeoff, or any selected confirmatory Freq-HRL algorithm.

### mujoco_v14_6_conservative_transfer_screen

The v14.6 development screen made lower-to-upper transfer function preserving and excluded untrained checkpoints. Every candidate exactly matched its paired control in selected parameters, latent-policy traces, executed-action traces, rewards, and returns. Strengths at or above 0.075 reduced both registered lower-LF diagnostics in all 15 conditions, but every such arm failed the upper-HF budget in all five Hopper conditions and passed only 10 of 15 complete gates. Because policy parameters and environment behavior were identical across arms, this is a responsibility-coordinate diagnostic rather than evidence of improved learned control.

Forbidden: MuJoCo v14.6 validates learned behavior-safe Freq-HRL, improves control performance, or supplies a selected algorithm for confirmatory testing.

### mujoco_v14_7_joint_learned_projection_preflight

The v14.7 single-replicate preflight validated pathwise projection calibration and confirmed that learned arms changed actor behavior, but the learned arms used a behavior-robust checkpoint selector while the comparator used mean reward. The result is excluded from algorithm selection.

Forbidden: MuJoCo v14.7 supports reward improvement, learned frequency separation, no-tradeoff behavior, or any confirmatory candidate.

### mujoco_v14_8_latent_matched_preflight

The v14.8 single-replicate HalfCheetah preflight removed the v14.7 comparator and actor-identity confounds. Projection-only calibration was pathwise exact, but no learned arm combined reward preservation with latent lower- and upper-frequency improvement. Equal upper and lower dual rates operated at incompatible cost scales.

Forbidden: MuJoCo v14.8 supports a learned no-tradeoff result, cross-task generality, reward improvement, or confirmatory evidence.

### mujoco_v14_9_asymmetric_feasibility_preflight

The v14.9 HalfCheetah development extension found that one asymmetric arm improved mean return in four modes and reduced latent upper-HF power, but worsened every mean lower-frequency endpoint. The audit showed that stochastic rollout-action constraints did not match deterministic deployment actions.

Forbidden: MuJoCo v14.9 supports joint learned frequency separation, no-tradeoff behavior, cross-task generality, confirmatory reward improvement, or a submission-ready selected algorithm.

### mujoco_v14_10_deployment_aligned_preflight

The source-bound v14.10 preflight validated deterministic deployment-frequency gradients, projection calibration, provenance, and held-out-grid integrity. Every learned arm fell back to the initial checkpoint because single-step corrections were smaller than the registered targets.

Forbidden: MuJoCo v14.10 supports learned frequency separation, reward improvement, no-tradeoff behavior, cross-task generality, confirmatory evidence, or a selected algorithm.

### mujoco_v14_11_iterative_projection_preflight

The source-bound v14.11 preflight showed that iterative deterministic projection increased same-batch frequency reduction without surrogate reward-budget violations. Every learned arm nevertheless selected the fallback because pooled improvements did not satisfy worst-condition reward and frequency ranks.

Forbidden: MuJoCo v14.11 supports an accepted learned checkpoint, reward improvement, no-tradeoff behavior, cross-task generality, confirmatory evidence, or a selected algorithm.

### mujoco_v14_12_groupwise_robust_preflight

The source-bound v14.12 preflight preserved four rollout groups and zero per-group surrogate reward-budget violations, but no arm met every paired reward floor and frequency target at an eligible learned checkpoint.

Forbidden: MuJoCo v14.12 supports an accepted learned checkpoint, reward improvement, no-tradeoff behavior, cross-task generality, confirmatory evidence, or a selected algorithm.

### mujoco_v14_13_anchor_replay_trust_preflight

The source-bound v14.13 preflight validated exact projection calibration, frozen anchor replay, and finite-budget PPO trust regions. All learned arms selected the fallback because trained checkpoints still failed unseen-seed closed-loop frequency endpoints.

Forbidden: MuJoCo v14.13 supports an accepted learned checkpoint, reward improvement, learned frequency separation, statistical evidence, confirmatory evidence, or a selected algorithm.

### mujoco_v14_14_closed_loop_actor_guard_preflight

The source-bound v14.14 preflight executed the independent closed-loop guard but accepted zero effective actor updates. It identified a feasible-set maintenance rule incorrectly applied at an infeasible starting point.

Forbidden: MuJoCo v14.14 supports an accepted learned checkpoint, reward improvement, learned frequency separation, statistical evidence, confirmatory evidence, or a selected algorithm.

### mujoco_v14_15_closed_loop_restoration_filter_preflight

The source-bound v14.15 HalfCheetah preflight selected the restoration arm with reward tolerance 0.005, eight backtracks, and funnel multiplier 3. It accepted 22 effective joint actor updates, reduced 20 closed-loop frequency violations and continuous violation merit to zero under the reward floor, and passed all five held-out disturbance gates. This single-seed result authorizes a fresh multiseed development screen only.

Forbidden: MuJoCo v14.15 preflight alone supports statistically reliable reward improvement, robust frequency separation, no-tradeoff behavior, cross-environment generality, confirmatory evidence, or a submission-ready algorithm.

### mujoco_v14_15_restoration_multiseed_development_r2

The repaired v14.15 r2 development screen completed all 450 frozen cells across 15 fresh optimizer seeds and three MuJoCo environments. The preselected restoration arm was not ready for confirmation: only 8 of 45 environment-by-seed candidate cells passed the complete gate (HalfCheetah 0, Hopper 7, Walker2d 1), and the simultaneous primary family failed. Training often restored the independent guard paths, but checkpoint selection exposed poor transfer to disjoint paths, motivating broader frozen-state coverage and restoration-specific regularization.

Forbidden: MuJoCo v14.15 provides confirmatory evidence, establishes cross-environment learned frequency separation, supports a no-tradeoff claim, or is ready for submission as the final algorithm.

### mujoco_v14_16_crossed_restoration_mechanism_development_r5

The source-bound v14.16 development screen recovered and validated all 81 registered cells. The preregistered crossed-replay arm passed neither the engineering nor complete effect gate in any of nine environment-by-seed cells; eight cells retained the fallback checkpoint. The non-frozen pathwise arm was the best diagnostic variant but completed only the Hopper environment and 2 of 9 cell-level gates. These results reject binary reward-actor freezing and all-path hard feasibility as the next scalable mechanism.

Forbidden: MuJoCo v14.16 supports learned cross-environment frequency separation, a no-tradeoff claim, a statistically reliable improvement, or expansion of the preregistered primary arm to confirmation.

### legacy_c1_c9_matrix_snapshot

This snapshot may be used only to trace historical claim changes. Its individual rows require record-level re-adjudication before manuscript use.

Forbidden: The historical 1-of-9 count is the current authoritative paper conclusion.

### legacy_paper_diagnostics_snapshot

This file is retained as a historical diagnostic inventory, not as a manuscript evidence source.

Forbidden: A path test, three-seed surrogate interval, or mechanism diagnostic in the historical file establishes a confirmatory domain-general result.
