from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_wait_constraint_stage289 as spec
from scripts import submit_native_transit_wait_constraint_stage289_scheduleurm as submit


def rows():
    return [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p}
        for k in spec.source.keys("train") for p in spec.source.PROBES]


class WaitConstraintTest(unittest.TestCase):
    def test_constraint_native_fallback_raw_first_ties_and_feasibility(self):
        values = np.zeros((3, 16, 6, 4))
        values[0, :, :, 0] = [0, -4, -2, -9, 1, 0]
        values[0, :, :, 3] = [0, 2, -1, -2, -1, 0]
        values[2, :, :, 0] = [-2, 0, 1, 1, 1, 1]
        values[2, :, :, 3] = [1, 0, 1, 1, 1, 1]
        raw = np.array([0, 3, 1])
        data = {"raw_logits": -np.ones((3, 6)), "originals": np.array([0, 1, 1])}
        data["raw_logits"][0, 3] = 1
        data["raw_logits"][2, 1] = 1
        choices = spec.propose(data, values, raw, raw)
        np.testing.assert_array_equal(choices[2], [1, 3, 0])
        np.testing.assert_array_equal(choices[3], [2, 3, 1])
        relative = values.mean(axis=1)-values.mean(axis=1)[np.arange(3), raw][:, None]
        self.assertTrue(np.all(relative[np.arange(3), choices[3], 0] <= 0))
        self.assertTrue(np.all(relative[np.arange(3), choices[3], 3] <= 0))

    def test_matched_disjoint_folds_and_no_replicated_model_noise(self):
        for selection, evaluation in spec.FOLDS.values():
            self.assertEqual((len(selection), len(evaluation)), (16, 8))
            self.assertFalse(set(selection)&set(evaluation))
            self.assertEqual(set(selection)|set(evaluation), set(range(24)))
        rr = rows()
        choices = np.zeros((1, 4, len(rr)), int)
        choices[:, 2:] = 1
        values = np.zeros((len(rr), 8, 6, 4))
        values[:, :, 1] = -np.arange(1., 9.)[None, :, None]
        single = spec.evaluate(rr, choices, values)
        repeated = spec.evaluate(rr, np.repeat(choices, 16, axis=0), values)
        self.assertEqual(single, repeated)
        zero = repeated["comparisons"]["wait_constrained_vs_cost_only"]
        self.assertEqual(zero["mean"]["cost"], 0.)
        self.assertEqual(zero["conditional_MC_se"]["cost"], 0.)
        self.assertEqual(zero["prefix_CI95"]["cost"], [0., 0.])

    def test_train_only_roundtrip_primary_ledger_precedes_independent_bank(self):
        rr = rows()
        values = np.zeros((360, 24, 6, 4))
        values[..., 0] = [0, -2, -1, 1, 1, 1]
        values[..., 2] = values[..., 0]
        values[..., 3] = [0, .3, -.5, 1, 1, 1]
        data = {"raw_logits": -np.ones((360, 6)), "originals": np.zeros(360, int), "raw_choices": np.zeros(360, int)}
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), \
                patch.object(spec.source, "ROOT", Path(temp)):
            run = "toy289"
            path = spec.directory(run)
            parent = spec.source.directory(spec.SOURCE_RUN)
            spec.write_json(path/"preregistration.json", {"contract": spec.contract()})
            spec.write_json(parent/"freeze_raw/models.json", [{"key": [i], "reference": {}} for i in range(16)])
            (parent/"fit_raw").mkdir()
            np.savez_compressed(parent/"fit_raw/train_predictions.npz", chosen=np.zeros((16, 6, 4, 360), int))
            calls = []

            def load(indices):
                calls.append(indices)
                if indices == spec.FOLDS["primary"][1]:
                    frozen = spec.read(path/"report_raw/primary.json")
                    np.testing.assert_array_equal(np.asarray(frozen["choices"])[:, 3], 2)
                return values[:, indices].copy()

            def train_rows(run_name, role):
                self.assertEqual((run_name, role), (spec.SOURCE_RUN, "train"))
                return rr

            with patch.object(spec.source, "load_rows", side_effect=train_rows), \
                    patch.object(spec, "load_draws", side_effect=load), \
                    patch.object(spec.source.learning.base.base, "causal_data", return_value=data), \
                    patch.object(spec.source, "load_panels", side_effect=AssertionError("No full/held panels")), \
                    patch.object(spec.source, "fit", side_effect=AssertionError("No fit")), \
                    patch.object(spec.source, "collect", side_effect=AssertionError("No simulation")):
                spec.analyze(run)
                report = spec.read(path/"report/result.json")
                values[:, 16:, 2, 3] = .5
                spec.analyze(run)
                harmed = spec.read(path/"report/result.json")
            self.assertEqual(calls, [spec.FOLDS["primary"][0], spec.FOLDS["primary"][1]]*2)
            self.assertTrue(report["joint_retention_supported"])
            self.assertFalse(harmed["joint_retention_supported"])
            self.assertFalse(harmed["primary_joint_CI_passed"])
            self.assertFalse(harmed["primary_both_halves_passed"])
            self.assertEqual(report["held_prefixes_read"], 0)
            self.assertEqual(report["new_fits"], 0)
            self.assertEqual(report["new_native_episodes"], 0)
            primary = report["queries"]["primary"]["independent8"]["comparisons"]
            self.assertGreater(primary["cost_only_vs_raw"]["mean"]["wait_min"], 0.)
            self.assertLess(primary["wait_constrained_vs_raw"]["mean"]["wait_min"], 0.)

    def test_loader_opens_only_requested_train_chunks(self):
        key = spec.source.keys("train")[0]
        paths = []

        def read(path):
            paths.append(str(path))
            if path.name == "receipt.json":
                return {"software_qualified": True, "contract": spec.source.contract()}
            return [{"draw": d, "values": {str(a): [d, a, 0, 0] for a in range(6)}} for d in range(4)]

        with patch.object(spec.source, "keys", return_value=[key]) as keys, \
                patch.object(spec.source, "PROBES", (8,)), patch.object(spec, "read", side_effect=read):
            values = spec.load_draws((0, 1, 2, 3))
        keys.assert_called_once_with("train")
        self.assertEqual(values.shape, (1, 4, 6, 4))
        self.assertTrue(all("/train/" in p and "/chunk0/" in p for p in paths))

    def test_scheduler_cpu_nodes_and_metadata_input(self):
        task = submit.task_specification("toy289")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn(str(spec.directory("toy289")), task["stage_input_paths"])
        self.assertEqual(task["result_dir"], str(spec.directory("toy289")/"report"))

    def test_parent_gate_rejects_unfinished_stage288(self):
        receipt = {"scheduler": {"collection_status_counts": {"done": 3120},
            "collection_exit_code_counts": {"0": 3120},
            **{p: {"status": "done", "exit_code": 0} for p in ("fit", "report")}}}

        def read(path):
            if path.name == "completion_receipt.json":
                return receipt
            return {"contract": spec.source.contract(), "models_and_choices_frozen": True, "held_labels_read": False}

        with patch.object(spec, "read", side_effect=read):
            spec.qualify_parent()
            receipt["scheduler"]["report"]["status"] = "running"
            with self.assertRaisesRegex(ValueError, "Complete qualified Stage288"):
                spec.qualify_parent()


if __name__ == "__main__":
    unittest.main()
