import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from freq_hrl.experiments.mujoco.pilot_analysis import write_analysis


class MujocoPilotAnalysisTest(unittest.TestCase):
    @staticmethod
    def _write_cell(
        root: Path, method: str, optimizer_seed: int, returns: list[float]
    ) -> None:
        cell = (
            root / "cells" / "HalfCheetah-v5" / method
            / f"replicate_{optimizer_seed}"
        )
        cell.mkdir(parents=True)
        parameter_hash = ("a" if method == "freq_hrl" else "b") * 64
        checkpoint = cell / "checkpoint.pt"
        torch.save({
            "model_state_dict": {},
            "frozen_parameter_sha256": parameter_hash,
        }, checkpoint)
        file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        summary = {
            "environment": "HalfCheetah-v5",
            "method": method,
            "protocol_version": "unit_mujoco_v4",
            "source_identity_status": "verified",
            "code_revision": "c" * 40,
            "source_manifest_sha256": "d" * 64,
            "iterations": 1,
            "selected_checkpoint_iteration": 0,
            "validation_learning_gain": 1.0,
            "initial_validation_score": 0.0,
            "checkpoint_selection_score": 1.0,
            "frozen_parameter_sha256": parameter_hash,
            "checkpoint_file_sha256": file_hash,
            "capacity_ratio": 1.0,
        }
        (cell / "cell_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (cell / "training_history.json").write_text(json.dumps([
            {"iteration": -1, "checkpoint_selected": True},
            {"iteration": 0, "checkpoint_selected": True},
        ]), encoding="utf-8")
        rows = [{
            "seed": seed,
            "environment": "HalfCheetah-v5",
            "disturbance_mode": "standard",
            "method": method,
            "episode_return": value,
            "LowerLFDriftAbs": 0.1 if method == "freq_hrl" else 0.2,
            "protocol_valid": 1.0,
            "training_replicate_seed": optimizer_seed,
            "evaluation_role": "heldout_test",
            "protocol_version": "unit_mujoco_v4",
        } for seed, value in zip((101, 103), returns)]
        with (cell / "evaluation_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_full_audit_clusters_on_optimizer_replicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            for seed, offset in ((11, 0.0), (13, 1.0)):
                self._write_cell(root, "freq_hrl", seed, [5.0 + offset, 7.0 + offset])
                self._write_cell(root, "flat_ppo", seed, [2.0 + offset, 4.0 + offset])
            payload = write_analysis(
                root,
                root / "audit",
                expected_protocol_version="unit_mujoco_v4",
                environments=["HalfCheetah-v5"],
                methods=["freq_hrl", "flat_ppo"],
                optimizer_seeds=[11, 13],
                evaluation_seeds=[101, 103],
                disturbance_modes=["standard"],
            )
            self.assertEqual(payload["audit"]["status"], "valid")
            effect = next(
                row for row in payload["paired_effects"]
                if row["metric"] == "episode_return"
                and row["treatment"] == "freq_hrl"
                and row["control"] == "flat_ppo"
            )
            self.assertEqual(effect["n_common"], 2)
            self.assertEqual(effect["n_independent"], 2)
            self.assertAlmostEqual(effect["improvement_mean"], 3.0)
            self.assertTrue((root / "audit" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
