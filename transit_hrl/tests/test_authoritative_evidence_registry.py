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
        self.assertEqual(len(records), 22)
        by_id = {row["evidence_id"]: row for row in records}
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
            self.assertEqual(summary["record_count"], 22)
            self.assertEqual(summary["reportable_record_count"], 3)
            self.assertEqual(summary["positive_supported_record_count"], 1)
            self.assertEqual(summary["mixed_or_negative_record_count"], 2)
            self.assertEqual(summary["development_record_count"], 17)
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
