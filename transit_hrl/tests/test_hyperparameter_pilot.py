import csv
import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.hyperparameter_pilot import (
    CANDIDATES_BY_ID,
    TUNING_PROTOCOL_VERSION,
    candidate_ids_for_mode,
    merge_hpo_cells,
    run_hpo_cell,
)
from freq_hrl.experiments.trading.metrics import SELECTION_OBJECTIVE_VERSION


class HyperparameterPilotTest(unittest.TestCase):
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
        self.assertTrue(all(
            row["evaluation_role"] == "tuning_validation"
            for row in payload["tuning_rows"]
        ))

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
        self.assertEqual(merged["frozen_config"]["status"], "frozen_from_validation_only")


if __name__ == "__main__":
    unittest.main()
