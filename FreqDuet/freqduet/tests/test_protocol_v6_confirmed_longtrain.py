import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.audit_protocol_v6_confirmed_longtrain import (
    CONFIRMED_MAIN,
    EXPECTED_CONFIGS,
    EXPECTED_EVAL_SEEDS,
    EXPECTED_TRAIN_SEEDS,
    MATCHED_CONTEXT,
    PARENT_PRIMARY,
    evaluate_confirmed_longtrain,
)


class ProtocolV6ConfirmedLongtrainTest(unittest.TestCase):
    def make_artifacts(self, root: Path):
        parent = root / "parent_gate.json"
        parent.write_text(json.dumps({
            "gate_version": "freqduet-v8-compact-primary-confirmation-v1",
            "status": "primary_confirmed",
            "primary_claim_eligible": True,
            "primary": PARENT_PRIMARY,
            "confirmation_design": {
                "confirmation_train_seeds": [809, 827],
                "confirmation_eval_seeds": [44011, 44017],
            },
            "primary_result": {
                "matrix_provenance": {
                    "source_fingerprint_sha256": "a" * 64,
                    "scenario_contract_sha256": "b" * 64,
                },
            },
        }))
        parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
        aggregate = root / "aggregate"
        aggregate.mkdir()
        (aggregate / "matrix_manifest.json").write_text(json.dumps({
            "stage": "confirmation",
            "independent_confirmation": True,
            "configs": EXPECTED_CONFIGS,
            "train_seeds": EXPECTED_TRAIN_SEEDS,
            "eval_seeds": EXPECTED_EVAL_SEEDS,
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "expected_rollouts": 256,
            "run_source_fingerprint": {"sha256": "a" * 64},
            "scenario_contract": {"sha256": "b" * 64},
            "run_git_provenance": {
                "commit": "c" * 40,
                "tracked_dirty": False,
            },
        }))
        rows = []
        for config in EXPECTED_CONFIGS:
            for train_seed in EXPECTED_TRAIN_SEEDS:
                for eval_seed in EXPECTED_EVAL_SEEDS:
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "headway_cv": (
                            0.18 if config == CONFIRMED_MAIN else 0.22),
                    })
        pd.DataFrame(rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        return aggregate, parent, parent_sha

    @staticmethod
    def base_result(*, journey_ci_high: float = 0.10,
                    cv_ci_high: float = -0.01):
        return {
            "status": "unique_pass",
            "candidate_results": [{
                "candidate": CONFIRMED_MAIN,
                "headway_cv_delta_ci_high": cv_ci_high,
                "journey_delta_ci_high": journey_ci_high,
            }],
        }

    def test_all_locked_longtrain_gates_control_the_claim(self):
        with TemporaryDirectory() as tmp:
            aggregate, parent, parent_sha = self.make_artifacts(Path(tmp))
            with patch(
                "scripts.audit_protocol_v6_confirmed_longtrain."
                "evaluate_selection",
                return_value=self.base_result(),
            ):
                result = evaluate_confirmed_longtrain(
                    aggregate,
                    parent_gate_path=parent,
                    expected_parent_gate_sha256=parent_sha,
                )

        self.assertEqual(result["status"], "longtrain_confirmed")
        self.assertTrue(result["longtrain_claim_eligible"])
        self.assertEqual(
            result["headway_cv_negative_train_seed_fraction"], 1.0)
        self.assertTrue(all(result["lineage_checks"].values()))
        self.assertTrue(all(result["longtrain_gates"].values()))

    def test_journey_noninferiority_ci_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, parent, parent_sha = self.make_artifacts(Path(tmp))
            with patch(
                "scripts.audit_protocol_v6_confirmed_longtrain."
                "evaluate_selection",
                return_value=self.base_result(journey_ci_high=0.16),
            ):
                result = evaluate_confirmed_longtrain(
                    aggregate,
                    parent_gate_path=parent,
                    expected_parent_gate_sha256=parent_sha,
                )

        self.assertEqual(result["status"], "longtrain_not_confirmed")
        self.assertFalse(result["longtrain_claim_eligible"])
        self.assertFalse(
            result["longtrain_gates"]["journey_ci_is_noninferior"])

    def test_parent_gate_hash_mismatch_fails_before_effect_audit(self):
        with TemporaryDirectory() as tmp:
            aggregate, parent, _ = self.make_artifacts(Path(tmp))
            with self.assertRaisesRegex(ValueError, "lineage checks failed"):
                evaluate_confirmed_longtrain(
                    aggregate,
                    parent_gate_path=parent,
                    expected_parent_gate_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
