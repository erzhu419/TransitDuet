import csv
import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.full_method_hpo import (
    ALL_VARIANT_IDS,
    CAPACITY_REFERENCE_METHOD_CONTRACT,
    CANDIDATES_BY_ID,
    EXECUTION_TIMELINE_CONTRACT,
    FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
    FULL_METHOD_TUNING_PROTOCOL_VERSION,
    VARIANTS_BY_ID,
    candidate_ids_for_variant,
    canonical_full_method_parameter_count,
    effective_parameters_for_variant,
    merge_hpo_cells,
    run_hpo_cell,
)
from freq_hrl.experiments.trading.metrics import SELECTION_OBJECTIVE_VERSION
from freq_hrl.experiments.trading.ppo_actor_critic import (
    FULL_METHOD_IMPLEMENTATION_VERSION,
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
)


class FullMethodHPOTest(unittest.TestCase):
    def test_equal_search_budget_and_registered_variant_families(self):
        counts = {
            variant_id: len(candidate_ids_for_variant(variant_id))
            for variant_id in ALL_VARIANT_IDS
        }
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual(len(ALL_VARIANT_IDS), 10)
        self.assertEqual(
            canonical_full_method_parameter_count(assets=2),
            39445,
        )

    def test_ablation_effective_configs_change_only_registered_mechanism(self):
        candidate_id = candidate_ids_for_variant("freq_hrl_full_v4")[1]
        full = effective_parameters_for_variant("freq_hrl_full_v4", candidate_id)
        expected_changes = {
            "freq_hrl_no_promotion_v4": {
                "method_contract",
                "promotion_replan_cost",
            },
            "freq_hrl_no_hf_lower_v4": {
                "method_contract",
                "lower_hf_order_scale",
            },
            "freq_hrl_no_leakage_v4": {
                "method_contract",
                "leakage_scale",
                "lower_lf_constraint_coef",
                "lower_lf_constraint_target",
                "lower_lf_dual_lr",
                "lower_lf_objective_weight",
                "lower_lf_effect_filter_gain",
                "lower_lf_raw_recenter_scale",
            },
        }
        for variant_id, expected in expected_changes.items():
            with self.subTest(variant_id=variant_id):
                ablated = effective_parameters_for_variant(variant_id, candidate_id)
                changed = {
                    key for key in full
                    if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)

    def test_cell_uses_causal_contract_without_heldout_access(self):
        candidate_id = "freq_lr3e4_std05_activegate"
        payload = run_hpo_cell(
            candidate_id=candidate_id,
            variant_id="freq_hrl_full_v4",
            scenario="persistent_shift",
            training_replicate_seed=2026,
            train_seeds=[42],
            checkpoint_validation_seeds=[57721],
            tuning_validation_seeds=[68207],
            steps=24,
            assets=2,
            iterations=1,
        )
        summary = payload["cell_summary"]
        self.assertEqual(summary["heldout_test_seeds"], [])
        self.assertEqual(summary["heldout_test_access_status"], "not_loaded")
        self.assertEqual(
            summary["execution_timeline_contract"], EXECUTION_TIMELINE_CONTRACT
        )
        self.assertEqual(
            summary["capacity_reference_method_contract"],
            CAPACITY_REFERENCE_METHOD_CONTRACT,
        )
        self.assertEqual(summary["capacity_ratio"], 1.0)
        self.assertEqual(summary["parameter_count"], 39445)
        self.assertEqual(summary["hf_intervention_pair_count"], 1)
        self.assertTrue(
            payload["hf_intervention_rows"][0]["paired_exogenous_path_identity"]
        )
        self.assertEqual(
            payload["tuning_rows"][0]["evaluation_role"], "tuning_validation"
        )

    def test_cell_rejects_checkpoint_tuning_seed_overlap(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            run_hpo_cell(
                candidate_id="freq_lr1e4_std15_conservative",
                variant_id="freq_hrl_full_v4",
                scenario="persistent_shift",
                training_replicate_seed=2026,
                train_seeds=[42],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[57721],
                steps=8,
                assets=2,
                iterations=1,
            )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_merge_forces_ablation_to_inherit_full_candidate(self):
        variants = [
            "freq_hrl_full_v4",
            "freq_hrl_no_promotion_v4",
            "freq_hrl_no_hf_lower_v4",
            "freq_hrl_no_leakage_v4",
        ]
        candidates = candidate_ids_for_variant("freq_hrl_full_v4")[:2]
        scenario = "persistent_shift"
        replicates = [7, 11]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = []
            for variant_id in variants:
                variant = VARIANTS_BY_ID[variant_id]
                for candidate_index, candidate_id in enumerate(candidates):
                    for replicate in replicates:
                        directory = (
                            root / variant_id / candidate_id / str(replicate)
                        )
                        directory.mkdir(parents=True)
                        if variant_id == "freq_hrl_full_v4":
                            utility = 1.0 - 0.5 * candidate_index
                        else:
                            utility = 0.1 + 5.0 * candidate_index
                        hf_applicable = variant_id != "freq_hrl_no_hf_lower_v4"
                        summary = {
                            "variant_id": variant_id,
                            "scientific_role": variant.scientific_role,
                            "ablation_of": variant.ablation_of,
                            "candidate_id": candidate_id,
                            "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
                            "candidate_parameters": CANDIDATES_BY_ID[candidate_id].parameters,
                            "effective_parameters": effective_parameters_for_variant(
                                variant_id, candidate_id
                            ),
                            "trainer_family": variant.trainer_family,
                            "policy_mode": variant.policy_mode,
                            "method_contract": variant.method_contract,
                            "scenario": scenario,
                            "training_replicate_seed": replicate,
                            "rollout_seed_roots": [42],
                            "checkpoint_validation_seeds": [57721],
                            "tuning_validation_seeds": [68207],
                            "heldout_test_seeds": [],
                            "heldout_test_access_status": "not_loaded",
                            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
                            "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
                            "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
                            "learned_baseline_implementation_version": (
                                LEARNED_BASELINE_IMPLEMENTATION_VERSION
                            ),
                            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
                            "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
                            "capacity_reference_method_contract": (
                                CAPACITY_REFERENCE_METHOD_CONTRACT
                            ),
                            "volume_impact_bps": 10.0,
                            "code_revision": "a" * 40,
                            "source_manifest_sha256": "b" * 64,
                            "source_identity_status": "verified",
                            "steps": 24,
                            "assets": 2,
                            "iterations": 1,
                            "parameter_count": 100,
                            "capacity_actual_parameter_count": 100,
                            "capacity_target_parameter_count": 100,
                            "capacity_ratio": 1.0,
                            "selected_checkpoint_iteration": 0,
                            "validation_learning_gain": 0.01,
                            "hf_intervention_pair_count": 1 if hf_applicable else 0,
                            "cell_status": "valid",
                        }
                        (directory / "cell_summary.json").write_text(
                            json.dumps(summary), encoding="utf-8"
                        )
                        self._write_csv(directory / "tuning_rows.csv", [{
                            "seed": 68207,
                            "variant_id": variant_id,
                            "candidate_id": candidate_id,
                            "scenario": scenario,
                            "training_replicate_seed": replicate,
                            "evaluation_role": "tuning_validation",
                            "selection_utility": utility,
                            "promotion_gate_transition_count": 1,
                            "promotion_replan_count": (
                                1
                                if variant_id == "freq_hrl_full_v4"
                                and candidate_index == 0
                                else 0
                            ),
                        }])
                        if hf_applicable:
                            self._write_csv(
                                directory / "hf_intervention_rows.csv",
                                [{
                                    "seed": 68207,
                                    "variant_id": variant_id,
                                    "candidate_id": candidate_id,
                                    "scenario": scenario,
                                    "training_replicate_seed": replicate,
                                    "evaluation_role": (
                                        "tuning_validation_mechanism_diagnostic"
                                    ),
                                    "paired_exogenous_path_identity": True,
                                    "lower_hf_action_sensitivity": 0.1,
                                }],
                            )
                        else:
                            (directory / "hf_intervention_rows.csv").write_text(
                                "", encoding="utf-8"
                            )
                        directories.append(directory)
            merged = merge_hpo_cells(
                directories,
                expected_variant_ids=variants,
                expected_candidate_ids=candidates,
                expected_scenarios=[scenario],
                expected_replicate_seeds=replicates,
                top_k=1,
                stage="pilot",
            )
        selected = merged["frozen_config"]["selected"]
        full_candidate = selected["freq_hrl_full_v4"]["candidate_id"]
        self.assertEqual(full_candidate, candidates[0])
        for variant_id in variants[1:]:
            self.assertEqual(selected[variant_id]["candidate_id"], full_candidate)
            self.assertEqual(
                selected[variant_id]["selection_source_variant"],
                "freq_hrl_full_v4",
            )
        self.assertEqual(
            merged["summary"]["equal_search_budget_status"], "supported"
        )
        self.assertEqual(
            merged["summary"]["mechanism_activity_status"], "supported"
        )


if __name__ == "__main__":
    unittest.main()
