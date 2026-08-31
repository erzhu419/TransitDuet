import copy
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.authoritative_evidence_registry import (
    DEFAULT_REGISTRY,
    DEFAULT_REPOSITORY_ROOT,
    build_registry_outputs,
    load_registry,
    validate_registry,
)


class AuthoritativeEvidenceRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DEFAULT_REPOSITORY_ROOT
        cls.registry_path = cls.root / DEFAULT_REGISTRY

    def test_registered_snapshot_is_hash_verified_and_fail_closed(self):
        records = validate_registry(
            load_registry(self.registry_path), self.root
        )
        self.assertEqual(len(records), 46)
        by_id = {row["evidence_id"]: row for row in records}
        v17 = by_id["mujoco_v17_zero_dc_plan_development"]
        self.assertEqual(v17["facts"]["gate_counts"]["all_cell_gates"], 1)
        self.assertEqual(v17["facts"]["gate_counts"]["reward_noninferior"], 3)
        self.assertEqual(
            v17["facts"]["gate_counts"]["raw_lower_lf_reduction_vs_smooth"],
            9,
        )
        self.assertTrue(v17["facts"]["all_failed_attempts_exit_zero"])
        v17_1 = by_id["mujoco_v17_1_headroom_homotopy_development"]
        self.assertEqual(
            v17_1["facts"]["reward_noninferiority_counts"],
            {
                "headroom_exact": 0,
                "headroom_homotopy": 1,
                "headroom_homotopy_promotion_05": 1,
                "headroom_homotopy_promotion_10": 1,
            },
        )
        self.assertIsNone(
            v17_1["facts"]["selected_arm_for_fresh_multiseed"]
        )
        self.assertFalse(v17_1["facts"]["support_gate"])
        v17_2 = by_id["mujoco_v17_2_smooth_macro_gauge_development"]
        self.assertEqual(v17_2["facts"]["paired_path_count"], 360)
        self.assertEqual(
            v17_2["facts"]["frequency_gate_counts"]["alpha_005"],
            {"upper": 2, "lower": 0, "joint": 0, "bounded": 3},
        )
        self.assertIsNone(
            v17_2["facts"]["selected_alpha_for_leakage_active_multiseed"]
        )
        self.assertFalse(v17_2["facts"]["support_gate"])
        v17_3 = by_id[
            "mujoco_v17_3_audit_optimal_macro_gauge_development"
        ]
        self.assertEqual(v17_3["facts"]["paired_path_count"], 120)
        self.assertEqual(
            v17_3["facts"]["frequency_gate_counts"],
            {
                "upper_hf_reduction": 0,
                "lower_lf_reduction": 2,
                "joint_merit_reduction": 2,
            },
        )
        self.assertFalse(
            v17_3["facts"]["eligible_for_leakage_active_multiseed"]
        )
        self.assertFalse(v17_3["facts"]["support_gate"])
        v17_4 = by_id[
            "mujoco_v17_4_streaming_audit_projection_development"
        ]
        self.assertEqual(v17_4["facts"]["paired_path_count"], 120)
        self.assertEqual(
            v17_4["facts"]["frequency_gate_counts"],
            {
                "upper_hf_absolute_budget": 3,
                "lower_lf_absolute_budget": 1,
                "upper_budget_feasibility": 2,
                "lower_lf_reduction": 3,
                "joint_merit_reduction": 3,
            },
        )
        self.assertFalse(
            v17_4["facts"]["eligible_for_streaming_projection_multiseed"]
        )
        self.assertFalse(v17_4["facts"]["support_gate"])
        v17_5 = by_id[
            "mujoco_v17_5_feasibility_diagnostic_development"
        ]
        self.assertEqual(
            v17_5["facts"]["endpoint_improvement_counts"],
            {
                "episode_return": 3,
                "upper_hf_power": 0,
                "lower_lf_drift": 1,
                "lower_budget_violation": 1,
                "joint_budget_feasible_rate": 0,
            },
        )
        self.assertEqual(
            v17_5["facts"]["legacy_replay_exact_environment_count"], 3
        )
        self.assertFalse(v17_5["facts"]["support_gate"])
        v17_6 = by_id[
            "mujoco_v17_6_full_horizon_oracle_development"
        ]
        self.assertEqual(v17_6["facts"]["path_count"], 120)
        self.assertEqual(
            v17_6["facts"]["overall"]["recoverable_path_count"], 81
        )
        self.assertEqual(
            v17_6["facts"]["overall"]["oracle_infeasible_path_count"],
            7,
        )
        self.assertEqual(v17_6["facts"]["actor_floor_case_count"], 7)
        self.assertTrue(
            v17_6["facts"]["eligible_for_causal_router_rebuild"]
        )
        self.assertTrue(
            v17_6["facts"]["eligible_for_actor_feasibility_rebuild"]
        )
        self.assertFalse(v17_6["facts"]["support_gate"])
        v17_8 = by_id[
            "mujoco_v17_8_causal_fir_distillation_development"
        ]
        self.assertEqual(v17_8["facts"]["path_count"], 120)
        self.assertEqual(v17_8["facts"]["grouped_seed_fold_count"], 8)
        self.assertEqual(
            v17_8["facts"]["selected_recovered_failure_count"], 7
        )
        self.assertEqual(
            v17_8["facts"]["diagnostic_gain_one_recovered_failure_count"],
            58,
        )
        self.assertFalse(v17_8["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_8["facts"]["support_gate"])
        v17_9 = by_id["mujoco_v17_9_prefix_hpf_fir_development"]
        self.assertEqual(
            v17_9["facts"]["selected_recovered_failure_count"], 48
        )
        self.assertEqual(
            v17_9["facts"]
            ["selected_preserved_baseline_feasible_path_count"],
            32,
        )
        self.assertEqual(
            v17_9["facts"]["recovered_failures_by_environment"],
            {"HalfCheetah-v5": 40, "Hopper-v5": 0, "Walker2d-v5": 8},
        )
        self.assertFalse(v17_9["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_9["facts"]["support_gate"])
        v17_10 = by_id[
            "mujoco_v17_10_horizon_reservoir_fir_development"
        ]
        self.assertEqual(
            v17_10["facts"]["selected_recovered_failure_count"], 48
        )
        self.assertEqual(
            v17_10["facts"]["largest_reservoir_recovered_failure_count"],
            63,
        )
        self.assertEqual(
            v17_10["facts"]["largest_reservoir_valid_path_count"], 113
        )
        self.assertFalse(v17_10["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_10["facts"]["support_gate"])
        v17_11 = by_id[
            "mujoco_v17_11_fractional_reservoir_fir_development"
        ]
        self.assertEqual(
            v17_11["facts"]["selected_recovered_failure_count"], 62
        )
        self.assertEqual(
            v17_11["facts"]["recovered_failures_by_environment"],
            {"HalfCheetah-v5": 40, "Hopper-v5": 14, "Walker2d-v5": 8},
        )
        self.assertEqual(
            v17_11["facts"]
            ["best_diagnostic_hopper_recovered_failure_count"],
            16,
        )
        self.assertTrue(
            v17_11["facts"]["router_only_development_closed"]
        )
        self.assertFalse(v17_11["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_11["facts"]["support_gate"])
        v17_12 = by_id[
            "mujoco_v17_12_nearest_feasible_action_oracle_development"
        ]
        self.assertEqual(v17_12["facts"]["path_count"], 120)
        self.assertEqual(
            v17_12["facts"]["reference_feasible_path_count"], 113
        )
        self.assertEqual(v17_12["facts"]["actor_floor_path_count"], 7)
        self.assertEqual(
            v17_12["facts"]["frequency_target_feasible_path_count"], 120
        )
        self.assertAlmostEqual(
            v17_12["facts"]["actor_floor_total_action_rms_maximum"],
            0.008117855266084743,
        )
        self.assertEqual(v17_12["facts"]["server_target_count"], 7)
        self.assertTrue(
            v17_12["facts"]["causal_actor_adapter_authorized"]
        )
        self.assertFalse(v17_12["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_12["facts"]["support_gate"])
        v17_13 = by_id[
            "mujoco_v17_13_causal_actor_adapter_development"
        ]
        self.assertEqual(v17_13["facts"]["candidate_count"], 900)
        self.assertEqual(v17_13["facts"]["full_oracle_candidate_count"], 48)
        self.assertEqual(
            v17_13["facts"]["selected_actor_floor_recovered_path_count"],
            3,
        )
        self.assertEqual(
            v17_13["facts"]
            ["selected_reference_feasible_preserved_path_count"],
            113,
        )
        self.assertEqual(
            v17_13["facts"]["unexamined_aggressive_gain_values"],
            [1.5, 2.0],
        )
        self.assertFalse(v17_13["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_13["facts"]["support_gate"])
        v17_14 = by_id[
            "mujoco_v17_14_exhaustive_actor_oracle_development"
        ]
        self.assertEqual(
            v17_14["facts"]["combined_exact_candidate_count"], 900
        )
        self.assertEqual(v17_14["facts"]["passing_candidate_count"], 0)
        self.assertEqual(
            v17_14["facts"]["selected_actor_floor_recovered_path_count"],
            6,
        )
        self.assertEqual(
            v17_14["facts"]
            ["selected_reference_feasible_preserved_path_count"],
            113,
        )
        self.assertEqual(
            v17_14["facts"]["unresolved_disturbance_mode"], "ood_chirp"
        )
        self.assertTrue(
            v17_14["facts"]["frozen_linear_fir_grid_closed"]
        )
        self.assertFalse(v17_14["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v17_14["facts"]["support_gate"])
        v18_1 = by_id[
            "mujoco_v18_1_state_actor_dataset_development"
        ]
        self.assertEqual(v18_1["facts"]["path_count"], 120)
        self.assertEqual(v18_1["facts"]["done_task_count"], 120)
        self.assertEqual(
            v18_1["facts"]["trajectory_step_count_by_environment"],
            {
                "HalfCheetah-v5": 40000,
                "Hopper-v5": 3322,
                "Walker2d-v5": 6549,
            },
        )
        self.assertTrue(
            v18_1["facts"]["pretransition_causal_alignment_valid"]
        )
        self.assertFalse(v18_1["facts"]["support_gate"])
        v18_2 = by_id[
            "mujoco_v18_2_state_conditioned_actor_development"
        ]
        self.assertEqual(
            v18_2["facts"]["selected_actor_floor_recovered_path_count"],
            3,
        )
        self.assertEqual(
            v18_2["facts"]["selected_actor_floor_recovery_by_seed"],
            {
                "2802248628": {"recovered": 0, "total": 2},
                "294864529": {"recovered": 3, "total": 5},
            },
        )
        self.assertEqual(
            v18_2["facts"]["v17_14_actor_floor_recovered_path_count"],
            6,
        )
        self.assertFalse(
            v18_2["facts"]
            ["state_conditioning_improved_frozen_reused_panel"]
        )
        self.assertFalse(v18_2["facts"]["fresh_validation_paths_accessed"])
        self.assertFalse(v18_2["facts"]["support_gate"])
        v18_3 = by_id[
            "mujoco_v18_3_causal_joint_projection_development"
        ]
        self.assertEqual(
            v18_3["facts"]["selected_direct_joint_feasible_path_count"],
            120,
        )
        self.assertEqual(
            v18_3["facts"]["selected_actor_floor_recovered_path_count"],
            7,
        )
        self.assertGreater(
            v18_3["facts"]["selected_reference_correction_rms_maximum"],
            0.01,
        )
        self.assertEqual(
            v18_3["facts"]["prefix_ledger_valid_path_count"], 42
        )
        self.assertFalse(v18_3["facts"]["support_gate"])
        v18_4 = by_id[
            "mujoco_v18_4_receding_joint_projection_development"
        ]
        self.assertEqual(
            v18_4["facts"]["selected_direct_joint_feasible_path_count"],
            69,
        )
        self.assertEqual(
            v18_4["facts"][
                "selected_exact_oracle_joint_feasible_path_count"
            ],
            120,
        )
        self.assertEqual(
            v18_4["facts"]["offline_exact_online_direct_gap_path_count"],
            51,
        )
        self.assertEqual(
            v18_4["facts"]["selected_actor_floor_recovered_path_count"],
            2,
        )
        self.assertEqual(
            v18_4["facts"][
                "selected_prefix_budget_violation_step_count"
            ],
            40962,
        )
        self.assertFalse(v18_4["facts"]["support_gate"])
        self.assertEqual(
            by_id["mujoco_v16_2_macro_hold_gauge_development"]["facts"][
                "gate_counts"
            ]["all_cell_gates"],
            2,
        )
        self.assertTrue(
            by_id["mujoco_v12_responsibility_confirmatory"][
                "positive_claim_supported"
            ]
        )
        self.assertTrue(
            by_id["mujoco_v13_behavioral_confirmatory"][
                "manuscript_reportable"
            ]
        )
        self.assertFalse(
            by_id["mujoco_v13_behavioral_confirmatory"][
                "positive_claim_supported"
            ]
        )
        quant = by_id["quant_v74_matched_baseline_confirmatory"]
        self.assertEqual(
            quant["facts"]["primary_status_counts"],
            {
                "supported_improvement": 8,
                "supported_harm": 1,
                "inconclusive": 3,
            },
        )
        self.assertFalse(
            by_id["mujoco_v14_endpoint_aligned_screen"][
                "manuscript_reportable"
            ]
        )
        v14_1 = by_id["mujoco_v14_1_crossed_upper_pd_screen"]
        self.assertFalse(v14_1["manuscript_reportable"])
        self.assertEqual(
            v14_1["facts"]["gate_granularity"],
            "environment_by_disturbance_mode",
        )
        self.assertTrue(all(
            status["passed_gate_count"] == 0
            for status in v14_1["facts"]["arm_status"].values()
        ))
        v14_2 = by_id["mujoco_v14_2_physical_router_screen"]
        self.assertFalse(v14_2["manuscript_reportable"])
        self.assertEqual(
            v14_2["facts"]["maximum_complete_condition_count"],
            2,
        )
        v14_3 = by_id["mujoco_v14_3_partial_router_screen"]
        self.assertFalse(v14_3["manuscript_reportable"])
        self.assertEqual(
            v14_3["facts"]["maximum_complete_condition_count"],
            4,
        )
        self.assertEqual(
            v14_3["facts"][
                "best_arm_responsibility_improvement_condition_count"
            ],
            15,
        )
        v14_4 = by_id["mujoco_v14_4_router_homotopy_screen"]
        self.assertFalse(v14_4["manuscript_reportable"])
        self.assertEqual(
            v14_4["facts"]["maximum_complete_condition_count"],
            3,
        )
        self.assertEqual(
            v14_4["facts"][
                "fastest_ramp_reward_noninferiority_condition_count"
            ],
            10,
        )
        self.assertEqual(
            v14_4["facts"]["fastest_ramp_raw_condition_count"],
            5,
        )
        v14_5 = by_id["mujoco_v14_5_paired_anchor_screen"]
        self.assertFalse(v14_5["manuscript_reportable"])
        self.assertEqual(
            v14_5["facts"]["maximum_complete_condition_count"],
            5,
        )
        self.assertEqual(
            v14_5["facts"][
                "full_drift_arm_reward_noninferiority_condition_count"
            ],
            5,
        )
        self.assertFalse(
            v14_5["facts"]["all_arms_trained_checkpoint_gate_pass"]
        )
        v14_6 = by_id["mujoco_v14_6_conservative_transfer_screen"]
        self.assertFalse(v14_6["manuscript_reportable"])
        self.assertEqual(
            v14_6["facts"]["maximum_complete_condition_count"],
            10,
        )
        self.assertEqual(
            v14_6["facts"][
                "full_drift_arm_reward_noninferiority_condition_count"
            ],
            15,
        )
        self.assertEqual(
            v14_6["facts"]["full_drift_arm_upper_hf_condition_count"],
            10,
        )
        self.assertTrue(v14_6["facts"]["all_arms_exact_return_trace_pass"])
        self.assertTrue(v14_6["facts"]["all_arms_exact_parameter_hash_pass"])
        self.assertTrue(
            v14_6["facts"]["all_arms_trained_checkpoint_gate_pass"]
        )
        v14_7 = by_id["mujoco_v14_7_joint_learned_projection_preflight"]
        self.assertEqual(v14_7["integrity_status"], "comparator_confounded")
        self.assertEqual(v14_7["facts"]["completed_continuation_cells"], 7)
        v14_15 = by_id[
            "mujoco_v14_15_closed_loop_restoration_filter_preflight"
        ]
        self.assertEqual(v14_15["decision"], "expand_to_multiseed_screen")
        self.assertEqual(
            v14_15["facts"]["selected_arm"],
            "group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3",
        )
        self.assertTrue(v14_15["facts"]["calibration_pass"])
        self.assertEqual(v14_15["facts"]["arm_count"], 6)
        self.assertFalse(v14_15["manuscript_reportable"])
        v14_15_multiseed = by_id[
            "mujoco_v14_15_restoration_multiseed_development_r2"
        ]
        self.assertEqual(
            v14_15_multiseed["decision"],
            "candidate_not_ready_for_confirmation",
        )
        self.assertFalse(v14_15_multiseed["manuscript_reportable"])
        self.assertEqual(
            v14_15_multiseed["facts"]["complete_candidate_cells"], 8
        )
        self.assertEqual(
            v14_15_multiseed["facts"]["candidate_cell_count"], 45
        )
        self.assertEqual(
            v14_15_multiseed["facts"]["complete_cells_by_environment"],
            {
                "HalfCheetah-v5": 0,
                "Hopper-v5": 7,
                "Walker2d-v5": 1,
            },
        )
        self.assertFalse(
            by_id["legacy_paper_diagnostics_snapshot"][
                "manuscript_reportable"
            ]
        )
        v14_16 = by_id[
            "mujoco_v14_16_crossed_restoration_mechanism_development_r5"
        ]
        self.assertEqual(v14_16["decision"], "primary_mechanism_not_ready")
        self.assertFalse(v14_16["manuscript_reportable"])
        self.assertEqual(v14_16["facts"]["merged_cell_count"], 81)
        self.assertEqual(v14_16["facts"]["rerouted_success_count"], 15)
        self.assertEqual(
            v14_16["facts"]["primary_engineering_pass_count"], 0
        )
        self.assertEqual(
            v14_16["facts"]["primary_fallback_checkpoint_count"], 8
        )
        v14_29 = by_id[
            "mujoco_v14_29_restoration_portfolio_confirmatory"
        ]
        self.assertTrue(v14_29["positive_claim_supported"])
        self.assertEqual(v14_29["facts"]["supported_cell_count"], 47)
        self.assertEqual(
            v14_29["facts"]["supported_count_by_environment"],
            {
                "HalfCheetah-v5": 16,
                "Hopper-v5": 16,
                "Walker2d-v5": 15,
            },
        )
        self.assertEqual(v14_29["facts"]["router_selection_count"], 38)
        self.assertEqual(v14_29["facts"]["actor_selection_count"], 9)
        self.assertEqual(v14_29["facts"]["abstention_count"], 1)
        self.assertTrue(
            v14_29["facts"]["all_selected_router_traces_invariant"]
        )
        v15 = by_id["mujoco_v15_raw_policy_distillation_development"]
        self.assertFalse(v15["manuscript_reportable"])
        self.assertEqual(
            v15["decision"],
            "universal_raw_policy_distillation_not_supported",
        )
        self.assertEqual(
            v15["facts"]["candidate_counts"],
            {"v15": 12, "v15.1": 108, "v15.2": 216},
        )
        self.assertEqual(
            v15["facts"]["validation_supported_count_by_run"],
            {"v15": 1, "v15.1": 1, "v15.2": 1},
        )
        self.assertFalse(v15["facts"]["universal_supported"])
        v16 = by_id["mujoco_v16_gauge_training_development"]
        self.assertFalse(v16["manuscript_reportable"])
        self.assertEqual(
            v16["decision"],
            "training_time_gauge_preflight_not_supported",
        )
        self.assertEqual(v16["facts"]["training_cell_count"], 27)
        self.assertEqual(v16["facts"]["paired_analysis_cell_count"], 9)
        self.assertEqual(
            v16["facts"]["gate_counts"],
            {
                "exact_reconstruction": 9,
                "reward_noninferiority": 6,
                "canonical_frequency_reduction": 0,
                "latent_noninferiority_vs_joint": 4,
                "latent_constraint_improvement": 5,
            },
        )
        self.assertFalse(v16["facts"]["support_gate"])
        v16_1 = by_id["mujoco_v16_1_audit_gauge_paired_development"]
        self.assertFalse(v16_1["manuscript_reportable"])
        self.assertEqual(
            v16_1["decision"],
            "audit_gauge_paired_preflight_not_supported",
        )
        self.assertEqual(v16_1["facts"]["training_cell_count"], 27)
        self.assertEqual(v16_1["facts"]["paired_analysis_cell_count"], 9)
        self.assertEqual(
            v16_1["facts"]["gate_counts"],
            {
                "candidate_selected_trained_checkpoint": 1,
                "selection_constraints_feasible": 4,
                "reward_noninferiority": 9,
                "canonical_frequency_reduction": 1,
                "latent_noninferiority_vs_control": 9,
                "exact_reconstruction": 9,
                "adaptive_cutoff_active": 9,
            },
        )
        self.assertFalse(v16_1["facts"]["support_gate"])

    def test_hash_tampering_is_rejected(self):
        registry = copy.deepcopy(load_registry(self.registry_path))
        registry["records"][0]["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_registry(registry, self.root)

    def test_positive_disposition_requires_confirmatory_design(self):
        registry = copy.deepcopy(load_registry(self.registry_path))
        registry["records"][0]["evidence_stage"] = "development"
        with self.assertRaisesRegex(
            ValueError, "positive disposition lacks confirmatory support"
        ):
            validate_registry(registry, self.root)

    def test_build_outputs_preserves_positive_and_negative_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = build_registry_outputs(
                registry_path=self.registry_path,
                repository_root=self.root,
                output_dir=root / "results",
                md_output=root / "ledger.md",
            )
            self.assertEqual(summary["record_count"], 46)
            self.assertEqual(summary["reportable_record_count"], 4)
            self.assertEqual(summary["positive_supported_record_count"], 2)
            self.assertEqual(summary["mixed_or_negative_record_count"], 2)
            self.assertEqual(summary["development_record_count"], 40)
            self.assertTrue((root / "results" / "summary.json").is_file())
            self.assertTrue((root / "ledger.md").is_file())

    def test_tracked_ledger_is_generated_from_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root / "ledger.md"
            build_registry_outputs(
                registry_path=self.registry_path,
                repository_root=self.root,
                output_dir=root / "results",
                md_output=generated,
            )
            tracked = (
                self.root
                / "transit_hrl/md/freq_hrl_authoritative_claim_ledger_2026-08-09.md"
            )
            self.assertEqual(generated.read_bytes(), tracked.read_bytes())


if __name__ == "__main__":
    unittest.main()
