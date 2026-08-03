import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.reproducibility import (
    current_freq_hrl_source_manifest_sha256,
)
from freq_hrl.experiments.trading.full_method_hpo import (
    ALL_VARIANT_IDS,
    CANDIDATES_BY_ID,
    FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
    FULL_METHOD_TUNING_PROTOCOL_VERSION,
    candidate_ids_for_variant,
    effective_parameters_for_variant,
    frozen_config_sha256,
)
from freq_hrl.experiments.trading.metrics import (
    METRIC_CONTRACT_VERSION,
    SELECTION_OBJECTIVE_VERSION,
)
from freq_hrl.experiments.trading.ppo_actor_critic import (
    FULL_METHOD_IMPLEMENTATION_VERSION,
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    DEFAULT_EVAL_SEEDS,
    DEFAULT_SCENARIOS,
)
from scripts.full_method_confirmatory import (
    CONFIRMATORY_ANALYSIS_VERSION,
    CONFIRMATORY_PROTOCOL_VERSION,
    DEFAULT_CONFIRMATORY_REPLICATE_SEEDS,
    DEFAULT_FRESH_HELDOUT_TEST_SEEDS,
    FULL_VARIANT,
    MINIMUM_HELDOUT_TEST_SEEDS,
    MINIMUM_STANDARDIZED_EFFECT,
    PRACTICAL_THRESHOLD_RULE,
    _protocol_comparisons,
    build_confirmatory_checks,
    canonical_sha256,
    run_confirmatory_cell,
    validate_confirmatory_protocol,
)


class FullMethodConfirmatoryTest(unittest.TestCase):
    @staticmethod
    def _frozen_config() -> dict:
        selected = {}
        full_candidate = candidate_ids_for_variant(FULL_VARIANT)[0]
        for variant_id in ALL_VARIANT_IDS:
            candidate_id = (
                full_candidate
                if variant_id.startswith("freq_hrl_")
                else candidate_ids_for_variant(variant_id)[0]
            )
            selected[variant_id] = {
                "candidate_id": candidate_id,
                "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
                "candidate_parameters": CANDIDATES_BY_ID[candidate_id].parameters,
                "effective_parameters": effective_parameters_for_variant(
                    variant_id, candidate_id
                ),
                "selection_source_variant": (
                    FULL_VARIANT
                    if variant_id.startswith("freq_hrl_")
                    else variant_id
                ),
                "selection_rule": (
                    "validation_cluster_lcb"
                    if variant_id == FULL_VARIANT
                    or not variant_id.startswith("freq_hrl_")
                    else "inherited_full_method_one_factor_ablation"
                ),
                "learning_gate_status": "eligible",
                "mechanism_activity_status": (
                    "eligible" if variant_id == FULL_VARIANT else "not_applicable"
                ),
                "trained_checkpoint_fraction": 1.0,
                "validation_learning_gain_mean": 0.01,
            }
        return {
            "status": "frozen_from_validation_only",
            "stage": "final",
            "final_design_complete": True,
            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
            "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "source_identity_status": "verified",
            "code_revision": "a" * 40,
            "source_manifest_sha256": current_freq_hrl_source_manifest_sha256(),
            "heldout_test_access_status": "not_loaded",
            "heldout_test_seeds": [],
            "rollout_seed_roots": [42],
            "checkpoint_validation_seeds": [57721],
            "tuning_validation_seeds": [68207],
            "scenarios": list(DEFAULT_SCENARIOS),
            "training_replicate_seeds": [2026, 2039, 2053, 2063, 2081],
            "search_budget_candidates_per_variant": {
                variant_id: 6 for variant_id in ALL_VARIANT_IDS
            },
            "equal_search_budget": True,
            "ablation_selection_rule": (
                "inherit_full_candidate_then_disable_exactly_one_registered_mechanism"
            ),
            "execution_timeline_contract": "causal_post_trade_v3",
            "capacity_reference_method_contract": "full_freq_hrl_v4",
            "volume_impact_bps": 10.0,
            "selected": selected,
            "top_candidates": {},
        }

    @classmethod
    def _protocol(cls, frozen: dict, heldout=None) -> dict:
        comparisons = [{
            **comparison,
            "practical_effect_threshold": 0.01,
            "validation_cluster_delta_sample_sd": 0.05,
            "validation_independent_training_replicates": 5,
            "threshold_rule": PRACTICAL_THRESHOLD_RULE,
            "metric_floor": 0.001,
        } for comparison in _protocol_comparisons()]
        payload = {
            "status": "preregistered_from_validation_before_heldout",
            "protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
            "analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
            "hpo_tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
            "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            "frozen_config_sha256": frozen_config_sha256(frozen),
            "algorithm_code_revision": frozen["code_revision"],
            "algorithm_source_manifest_sha256": frozen["source_manifest_sha256"],
            "heldout_test_access_status_at_freeze": "not_loaded",
            "heldout_test_seeds": list(
                heldout or DEFAULT_FRESH_HELDOUT_TEST_SEEDS
            ),
            "checkpoint_validation_seeds": [57721],
            "tuning_validation_seeds": [68207],
            "confirmatory_replicate_seeds": list(
                DEFAULT_CONFIRMATORY_REPLICATE_SEEDS
            ),
            "scenarios": list(DEFAULT_SCENARIOS),
            "variant_ids": list(ALL_VARIANT_IDS),
            "selected": frozen["selected"],
            "training_budget": {
                "steps": 16,
                "assets": 2,
                "iterations": 1,
                "rollout_seed_roots": [42],
            },
            "independent_sampling_unit": "training_replicate_seed",
            "pair_keys": ["scenario", "training_replicate_seed", "seed"],
            "minimum_independent_replicates": 10,
            "minimum_standardized_effect_dz": MINIMUM_STANDARDIZED_EFFECT,
            "familywise_alpha": 0.05,
            "multiplicity_method": (
                "Holm-Bonferroni within preregistered family"
            ),
            "practical_threshold_rule": PRACTICAL_THRESHOLD_RULE,
            "comparisons": comparisons,
        }
        payload["protocol_sha256"] = canonical_sha256(payload)
        return payload

    def test_fresh_training_and_test_seeds_do_not_reuse_old_protocol(self):
        self.assertFalse(
            set(DEFAULT_CONFIRMATORY_REPLICATE_SEEDS)
            & {2026, 2039, 2053, 2063, 2081, 2089, 2099, 2111, 2129, 2141}
        )
        self.assertFalse(
            set(DEFAULT_FRESH_HELDOUT_TEST_SEEDS) & set(DEFAULT_EVAL_SEEDS)
        )

    def test_protocol_hash_rejects_any_post_registration_edit(self):
        frozen = self._frozen_config()
        protocol = self._protocol(frozen)
        audit = validate_confirmatory_protocol(protocol, frozen=frozen)
        self.assertEqual(audit["status"], "valid")
        protocol["familywise_alpha"] = 0.10
        with self.assertRaisesRegex(ValueError, "content drifted"):
            validate_confirmatory_protocol(protocol, frozen=frozen)

    def test_protocol_rejects_too_few_heldout_test_seeds(self):
        frozen = self._frozen_config()
        protocol = self._protocol(
            frozen,
            heldout=DEFAULT_FRESH_HELDOUT_TEST_SEEDS[
                : MINIMUM_HELDOUT_TEST_SEEDS - 1
            ],
        )
        with self.assertRaisesRegex(ValueError, "too few held-out"):
            validate_confirmatory_protocol(protocol, frozen=frozen)

    def test_confirmatory_cell_loads_only_fresh_heldout_role(self):
        frozen = self._frozen_config()
        protocol = self._protocol(frozen)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen_path = root / "frozen.json"
            protocol_path = root / "protocol.json"
            frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            payload = run_confirmatory_cell(
                protocol_path=protocol_path,
                frozen_config_path=frozen_path,
                variant_id=FULL_VARIANT,
                scenario="persistent_shift",
                training_replicate_seed=DEFAULT_CONFIRMATORY_REPLICATE_SEEDS[0],
            )
        summary = payload["cell_summary"]
        self.assertEqual(summary["tuning_validation_seeds_loaded"], [])
        self.assertEqual(
            summary["heldout_test_seeds"],
            list(DEFAULT_FRESH_HELDOUT_TEST_SEEDS),
        )
        self.assertEqual(summary["capacity_ratio"], 1.0)
        self.assertEqual(payload["heldout_rows"][0]["evaluation_role"], "heldout_test")
        self.assertEqual(
            len(payload["hf_intervention_rows"]),
            MINIMUM_HELDOUT_TEST_SEEDS,
        )

    def test_paired_cluster_holm_supports_consistent_large_effects(self):
        frozen = self._frozen_config()
        protocol = self._protocol(frozen)
        rows = []
        for replicate_index, replicate in enumerate(
            DEFAULT_CONFIRMATORY_REPLICATE_SEEDS
        ):
            for variant_index, variant_id in enumerate(ALL_VARIANT_IDS):
                is_full = variant_id == FULL_VARIANT
                rows.append({
                    "variant_id": variant_id,
                    "scenario": "persistent_shift",
                    "training_replicate_seed": replicate,
                    "seed": DEFAULT_FRESH_HELDOUT_TEST_SEEDS[0],
                    "total_return": (
                        2.0 + 0.01 * replicate_index
                        if is_full else 0.1 * variant_index
                    ),
                    "LowerLFDrift": (
                        0.1 + 0.001 * replicate_index
                        if is_full else 2.0 + 0.05 * variant_index
                    ),
                })
        checks = build_confirmatory_checks(rows, protocol)
        self.assertEqual(len(checks), len(_protocol_comparisons()))
        superiority = [
            row for row in checks if row["test_type"] == "superiority"
        ]
        self.assertTrue(all(row["n_independent"] == 10 for row in checks))
        self.assertTrue(all(row["status"] == "supported" for row in superiority))
        noninferiority = [
            row for row in checks if row["test_type"] == "noninferiority"
        ]
        self.assertEqual(noninferiority[0]["status"], "supported")


if __name__ == "__main__":
    unittest.main()
