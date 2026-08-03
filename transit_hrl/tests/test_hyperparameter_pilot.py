import csv
import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.hyperparameter_pilot import (
    ALL_POLICY_MODES,
    CANDIDATES_BY_ID,
    TUNING_PROTOCOL_VERSION,
    candidate_ids_for_mode,
    frozen_config_sha256,
    merge_hpo_cells,
    run_hpo_cell,
    validate_frozen_config,
)
from freq_hrl.experiments.trading.metrics import SELECTION_OBJECTIVE_VERSION
from freq_hrl.experiments.trading.ppo_actor_critic import (
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    DEFAULT_SCENARIOS,
)


class HyperparameterPilotTest(unittest.TestCase):
    @staticmethod
    def _valid_freeze():
        selected = {}
        for mode in ALL_POLICY_MODES:
            candidate_id = candidate_ids_for_mode(mode)[0]
            candidate = CANDIDATES_BY_ID[candidate_id]
            selected[mode] = {
                "candidate_id": candidate_id,
                "candidate_family": candidate.family,
                "parameters": candidate.parameters,
                "learning_gate_status": "eligible",
                "trained_checkpoint_fraction": 1.0,
                "validation_learning_gain_mean": 0.01,
            }
        return {
            "status": "frozen_from_validation_only",
            "stage": "final",
            "final_design_complete": True,
            "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "source_identity_status": "verified",
            "code_revision": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "heldout_test_access_status": "not_loaded",
            "heldout_test_seeds": [],
            "checkpoint_validation_seeds": [57721],
            "tuning_validation_seeds": [68207],
            "scenarios": list(DEFAULT_SCENARIOS),
            "training_replicate_seeds": [7, 11, 13, 17, 19],
            "search_budget_candidates_per_policy": {
                mode: 8 for mode in ALL_POLICY_MODES
            },
            "selected": selected,
        }

    def test_frozen_config_validator_rejects_drift_and_test_access(self):
        payload = self._valid_freeze()
        audit = validate_frozen_config(payload)
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["sha256"], frozen_config_sha256(payload))
        self.assertEqual(len(audit["sha256"]), 64)
        self.assertEqual(audit["code_revision"], "a" * 40)

        contaminated = json.loads(json.dumps(payload))
        contaminated["heldout_test_seeds"] = [31415]
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_frozen_config(contaminated)

        drifted = json.loads(json.dumps(payload))
        drifted["selected"]["freq_hrl"]["parameters"]["learning_rate"] = 9.9
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_frozen_config(drifted)

        unregistered = json.loads(json.dumps(payload))
        unregistered["source_identity_status"] = "unregistered_local"
        with self.assertRaisesRegex(ValueError, "source identity"):
            validate_frozen_config(unregistered)

    def test_equal_candidate_search_budget_by_algorithm_family(self):
        self.assertEqual(len(candidate_ids_for_mode("freq_hrl")), 8)
        self.assertEqual(len(candidate_ids_for_mode("flat_ppo")), 8)
        self.assertEqual(len(candidate_ids_for_mode("flat_sac")), 8)
        self.assertEqual(len(candidate_ids_for_mode("flat_td3")), 8)

    def test_cell_never_loads_confirmatory_test_seeds(self):
        payload = run_hpo_cell(
            candidate_id=candidate_ids_for_mode("freq_hrl")[0],
            policy_mode="freq_hrl",
            scenario="persistent_shift",
            training_replicate_seed=7,
            train_seeds=[42],
            checkpoint_validation_seeds=[57721],
            tuning_validation_seeds=[68207],
            steps=16,
            assets=2,
            iterations=1,
        )
        summary = payload["cell_summary"]
        self.assertEqual(summary["heldout_test_seeds"], [])
        self.assertEqual(summary["heldout_test_access_status"], "not_loaded")
        self.assertEqual(summary["evaluation_role"], "tuning_validation")
        self.assertEqual(summary["selection_objective_version"], SELECTION_OBJECTIVE_VERSION)
        self.assertEqual(summary["source_identity_status"], "unregistered_local")
        self.assertTrue(all(
            row["evaluation_role"] == "tuning_validation"
            for row in payload["tuning_rows"]
        ))

    def test_cell_rejects_staged_source_manifest_drift(self):
        with self.assertRaisesRegex(RuntimeError, "staged source manifest mismatch"):
            run_hpo_cell(
                candidate_id=candidate_ids_for_mode("freq_hrl")[0],
                policy_mode="freq_hrl",
                scenario="persistent_shift",
                training_replicate_seed=7,
                train_seeds=[42],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[68207],
                steps=8,
                assets=2,
                iterations=1,
                code_revision="a" * 40,
                expected_source_manifest_sha256="0" * 64,
            )

    def test_cell_rejects_checkpoint_tuning_seed_overlap(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            run_hpo_cell(
                candidate_id=candidate_ids_for_mode("freq_hrl")[0],
                policy_mode="freq_hrl",
                scenario="persistent_shift",
                training_replicate_seed=7,
                train_seeds=[42],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[57721],
                steps=8,
                assets=2,
                iterations=1,
            )

    def test_merge_selects_per_policy_using_replicate_cluster_lcb(self):
        candidates = candidate_ids_for_mode("freq_hrl")[:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = []
            for candidate_index, candidate_id in enumerate(candidates):
                for replicate in (7, 11):
                    directory = root / candidate_id / str(replicate)
                    directory.mkdir(parents=True)
                    utility = 0.2 - 0.1 * candidate_index + 0.001 * replicate
                    summary = {
                        "candidate_id": candidate_id,
                        "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
                        "candidate_parameters": CANDIDATES_BY_ID[candidate_id].parameters,
                        "policy_mode": "freq_hrl",
                        "scenario": "persistent_shift",
                        "training_replicate_seed": replicate,
                        "checkpoint_validation_seeds": [57721],
                        "tuning_validation_seeds": [68207],
                        "heldout_test_seeds": [],
                        "cell_status": "valid",
                        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
                        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
                        "learned_baseline_implementation_version": (
                            LEARNED_BASELINE_IMPLEMENTATION_VERSION
                        ),
                        "selected_checkpoint_iteration": 0,
                        "validation_learning_gain": 0.01,
                    }
                    (directory / "cell_summary.json").write_text(json.dumps(summary))
                    with (directory / "tuning_rows.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "policy_mode",
                            "candidate_id",
                            "scenario",
                            "training_replicate_seed",
                            "evaluation_role",
                            "selection_utility",
                        ])
                        writer.writeheader()
                        writer.writerow({
                            "policy_mode": "freq_hrl",
                            "candidate_id": candidate_id,
                            "scenario": "persistent_shift",
                            "training_replicate_seed": replicate,
                            "evaluation_role": "tuning_validation",
                            "selection_utility": utility,
                        })
                    directories.append(directory)
            merged = merge_hpo_cells(
                directories,
                expected_policy_modes=["freq_hrl"],
                expected_candidate_ids=candidates,
                expected_scenarios=["persistent_shift"],
                expected_replicate_seeds=[7, 11],
                top_k=1,
                stage="final",
            )
        self.assertEqual(merged["summary"]["matrix_coverage_status"], "complete")
        self.assertEqual(merged["summary"]["heldout_test_access_count"], 0)
        self.assertEqual(
            merged["frozen_config"]["selected"]["freq_hrl"]["candidate_id"],
            candidates[0],
        )
        self.assertEqual(
            merged["frozen_config"]["status"],
            "not_freezable_incomplete_final_design",
        )
        self.assertFalse(merged["frozen_config"]["final_design_complete"])


if __name__ == "__main__":
    unittest.main()
