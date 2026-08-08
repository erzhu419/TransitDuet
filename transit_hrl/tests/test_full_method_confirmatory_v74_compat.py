import unittest
import json
import tempfile
from pathlib import Path

from freq_hrl.experiments.reproducibility import derive_seed
from freq_hrl.experiments.trading import full_method_confirmatory_v732 as v732
from freq_hrl.experiments.trading import full_method_confirmatory_plan_v74 as plan
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo
from scripts import full_method_confirmatory_v74_compat as confirm


class FullMethodConfirmatoryV74CompatTest(unittest.TestCase):
    @staticmethod
    def _frozen() -> dict:
        candidate = "v73_balanced_strict"
        return {
            "selected": {
                "freq_hrl_full_v7": {
                    "candidate_id": candidate,
                    "effective_parameters": hpo.effective_parameters_for_variant(
                        "freq_hrl_full_v7", candidate
                    ),
                }
            },
            "rollout_seed_roots": [42],
            "promotion_calibration_seeds": [43],
            "checkpoint_validation_seeds": [44],
            "tuning_validation_seeds": [45],
            "training_replicate_seeds": list(
                hpo.DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS
            ),
            "steps": 12,
            "assets": 2,
            "iterations": 1,
            "confirmatory_plan_version": plan.CONFIRMATORY_PLAN_VERSION,
            "confirmatory_plan_sha256": plan.plan_sha256(),
        }

    def test_private_engine_uses_only_the_v74_plan(self):
        self.assertIs(confirm.engine.plan, plan)
        self.assertEqual(
            confirm.engine.CONFIRMATORY_PROTOCOL_VERSION,
            confirm.CONFIRMATORY_PROTOCOL_VERSION,
        )
        self.assertEqual(
            confirm.engine.DEFAULT_CONFIRMATORY_REPLICATES,
            plan.DEFAULT_CONFIRMATORY_REPLICATES,
        )
        self.assertEqual(len(confirm.runtime_sha256()), 64)
        self.assertNotEqual(
            v732.plan.CONFIRMATORY_PLAN_VERSION,
            plan.CONFIRMATORY_PLAN_VERSION,
        )

    def test_registered_roles_validate_against_v74_plan(self):
        frozen = self._frozen()
        heldout = confirm.engine._validate_confirmatory_roles(
            frozen,
            training_replicate_seed=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
            heldout_seeds=plan.DEFAULT_HELDOUT_SEEDS,
        )
        self.assertEqual(tuple(heldout), plan.DEFAULT_HELDOUT_SEEDS)

    def test_seed_namespace_is_versioned_without_touching_base_engine(self):
        value = confirm.engine.derive_seed(
            "freq_hrl_v732_confirmatory_optimizer", 81013
        )
        self.assertEqual(
            value,
            derive_seed("freq_hrl_v74_confirmatory_optimizer", 81013),
        )

    def test_frozen_training_runs_without_heldout_access(self):
        _, _, _, calibration, rows = confirm.engine._train_frozen_variant(
            frozen=self._frozen(),
            variant_id="freq_hrl_full_v7",
            training_replicate_seed=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
        )
        self.assertEqual(calibration["status"], "calibrated")
        self.assertTrue(rows)

    def test_cell_annotation_records_adapter_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_summary.json"
            path.write_text(json.dumps({"cell_status": "valid"}))
            confirm._annotate_cell(Path(tmp))
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["confirmatory_runtime_compatibility_version"],
                confirm.RUNTIME_COMPATIBILITY_VERSION,
            )
            self.assertEqual(
                payload["confirmatory_runtime_adapter_sha256"],
                confirm.runtime_sha256(),
            )

    def test_merge_rejects_unannotated_runtime_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_summary.json"
            path.write_text(json.dumps({"cell_status": "valid"}))
            with self.assertRaisesRegex(ValueError, "runtime provenance"):
                confirm.merge_confirmatory_cells(
                    [Path(tmp)],
                    expected_variant_ids=list(hpo.ALL_VARIANT_IDS),
                    expected_training_replicates=list(
                        plan.DEFAULT_CONFIRMATORY_REPLICATES
                    ),
                    expected_heldout_seeds=list(plan.DEFAULT_HELDOUT_SEEDS),
                )


if __name__ == "__main__":
    unittest.main()
