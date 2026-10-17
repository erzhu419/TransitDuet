from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_fleet_screen_stage290 as spec
from scripts import submit_native_transit_fleet_screen_stage290_scheduleurm as submit


def rows():
    return [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p}
        for k in spec.base.source.keys("train") for p in spec.base.source.PROBES]


class FleetScreenTest(unittest.TestCase):
    def test_three_disjoint_matched_banks_and_same_evaluation_role_swap(self):
        self.assertEqual(spec.QUERIES["primary"], (0, 1, 2))
        self.assertEqual(len(spec.QUERIES), 6)
        for p, s, e in spec.QUERIES.values():
            self.assertEqual({p, s, e}, {0, 1, 2})
            self.assertEqual([len(spec.BANKS[b]) for b in (p, s, e)], [8, 8, 8])
            self.assertFalse(set(spec.BANKS[p])&set(spec.BANKS[s]))
            self.assertFalse(set(spec.BANKS[p]+spec.BANKS[s])&set(spec.BANKS[e]))
        self.assertIn((1, 0, 2), spec.QUERIES.values())

    def test_screen_paired_raw_fallback_not_alternative_search(self):
        values = np.zeros((3, 8, 6, 4))
        values[..., 1] = np.arange(8)[None, :, None]+np.array([0, 1, -1, 0, -5, 0])[None, None]
        raw = np.zeros((2, 3), int)
        proposal = np.array([[1, 2, 0], [2, 1, 3]])
        chosen, accepted, zero = spec.screen(values, proposal, raw)
        np.testing.assert_array_equal(chosen, [[0, 2, 0], [2, 0, 3]])
        self.assertTrue(accepted[1, 2])
        self.assertTrue(zero[1, 2])
        self.assertFalse(accepted[0, 0])
        values[:, :4, 3, 1] += 1
        values[:, 4:, 3, 1] -= 1
        _, accepted, zero = spec.screen(values, proposal, raw)
        self.assertTrue(accepted[1, 2])
        self.assertFalse(zero[1, 2])

    def test_gate_contributions_and_zero_event_are_not_zero_risk(self):
        screening = np.zeros((3, 8, 6, 4))
        screening[..., 1] = [0, 1, -1, 0, -5, 0]
        raw = np.zeros((2, 3), int)
        proposal = np.array([[1, 2, 0], [2, 1, 3]])
        screened, accepted, zero = spec.screen(screening, proposal, raw)
        choices = np.stack((raw, raw, proposal, proposal, screened, proposal), axis=1)
        evaluation = screening.copy()
        evaluation[:, :, 3, 1] = 4
        evaluation[..., 0] = evaluation[..., 1]
        stats = spec.gate_diagnostics(choices, accepted, zero, evaluation)
        self.assertEqual(stats["proposed_changes"], 5)
        self.assertEqual(stats["counts"], {"accepted": 3, "rejected": 2, "accepted_zero_observed_fleet": 1})
        contribution = stats["evaluation_contributions_vs_raw"]
        self.assertGreater(contribution["accepted_zero_observed_fleet"]["fleet"], 0)
        before = spec.paired_values(evaluation, proposal, raw).mean(axis=(0, 1, 2))
        after = spec.paired_values(evaluation, screened, raw).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(list(contribution["accepted"].values()), after)
        np.testing.assert_allclose(np.array(list(contribution["accepted"].values()))+list(contribution["rejected"].values()), before)

    def test_six_arm_statistics_preserve_shared_model_noise(self):
        rr = rows()
        choices = np.zeros((1, 6, len(rr)), int)
        choices[:, 2:] = 1
        values = np.zeros((len(rr), 8, 6, 4))
        values[:, :, 1] = -np.arange(1., 9.)[None, :, None]
        options = {"arms": spec.ARMS, "pairs": spec.PAIRS, "seed": 290}
        single = spec.base.evaluate(rr, choices, values, **options)
        repeated = spec.base.evaluate(rr, np.repeat(choices, 16, axis=0), values, **options)
        self.assertEqual(single, repeated)
        zero = repeated["comparisons"]["independent_screen_vs_proposal8"]
        self.assertEqual(zero["conditional_MC_se"]["cost"], 0.)
        self.assertEqual(zero["prefix_CI95"]["cost"], [0., 0.])

    def test_train_only_roundtrip_both_freezes_and_exact_stage289_controls(self):
        rr = rows()
        values = np.zeros((360, 24, 6, 4))
        values[..., 0] = 2
        values[..., 2] = 2
        values[..., 3] = 1
        values[:, :, 0] = 0
        values[:, :, 2] = [-1, -.5, -.5, -.4]
        values[:, :8, 1] = [-2, -3, 1, -.5]
        values[:, 8:16, 1] = [-.5, 1, -1.5, -.5]
        values[:, 16:, 1] = [1.5, 3, -1.5, -.5]
        data = {"raw_logits": -np.ones((360, 6)), "originals": np.zeros(360, int), "raw_choices": np.zeros(360, int)}
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), \
                patch.object(spec.base, "ROOT", Path(temp)), patch.object(spec.base.source, "ROOT", Path(temp)):
            run = "toy290"
            path = spec.directory(run)
            parent = spec.base.source.directory(spec.base.SOURCE_RUN)
            spec.write_json(path/"preregistration.json", {"contract": spec.contract()})
            spec.write_json(parent/"freeze_raw/models.json", [{"key": [i], "reference": {}} for i in range(16)])
            (parent/"fit_raw").mkdir()
            np.savez_compressed(parent/"fit_raw/train_predictions.npz", chosen=np.zeros((16, 6, 4, 360), int))
            for name, (selection, _) in spec.base.FOLDS.items():
                old = np.stack([spec.base.propose(data, values[:, selection], data["raw_choices"], data["raw_choices"]) for _ in range(16)])
                spec.write_json(spec.base.directory(spec.SOURCE_RUN)/"report_raw"/(name+".json"), {"choices": old.tolist()})
            calls = []

            def load(indices):
                calls.append(indices)
                if indices == spec.BANKS[1]:
                    ledger = spec.read(path/"report_raw/primary_proposal.json")
                    np.testing.assert_array_equal(ledger["choices"], 1)
                if indices == spec.BANKS[2]:
                    ledger = spec.read(path/"report_raw/primary.json")
                    np.testing.assert_array_equal(np.asarray(ledger["choices"])[:, 4], 0)
                return values[:, indices].copy()

            def train_rows(run_name, role):
                self.assertEqual((run_name, role), (spec.base.SOURCE_RUN, "train"))
                return rr

            with patch.object(spec.base.source, "load_rows", side_effect=train_rows), \
                    patch.object(spec.base, "load_draws", side_effect=load), \
                    patch.object(spec.base.source.learning.base.base, "causal_data", return_value=data), \
                    patch.object(spec.base.source, "load_panels", side_effect=AssertionError("No full/held panels")), \
                    patch.object(spec.base.source, "fit", side_effect=AssertionError("No fit")), \
                    patch.object(spec.base.source, "collect", side_effect=AssertionError("No simulation")):
                spec.analyze(run)
                report = spec.read(path/"report/result.json")
                self.assertEqual(calls, list(spec.BANKS))
                self.assertEqual(len(report["queries"]), 6)
                self.assertTrue(report["stage289_control_replay_exact"])
                self.assertTrue(report["fleet_reduction_supported"])
                self.assertFalse(report["joint_retention_supported"])
                self.assertEqual((report["new_fits"], report["new_native_episodes"], report["held_prefixes_read"]), (0, 0, 0))
                old_path = spec.base.directory(spec.SOURCE_RUN)/"report_raw/primary.json"
                wrong = spec.read(old_path)
                wrong["choices"][0][3][0] = 5
                spec.write_json(old_path, wrong)
                with self.assertRaisesRegex(ValueError, "pooled16 choices differ"):
                    spec.analyze(run)

    def test_scheduler_resources_metadata_and_report_only_pull(self):
        task = submit.task_specification("toy290")
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 3072, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn(str(spec.directory("toy290")), task["stage_input_paths"])
        self.assertEqual(task["result_dir"], str(spec.directory("toy290")/"report"))

    def test_parent_gate_requires_completed_train_only_stage289(self):
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}, "held_prefixes_read": 0}

        def read(path):
            return receipt if path.name == "completion_receipt.json" else {"contract": spec.base.contract()}

        with patch.object(spec, "read", side_effect=read):
            spec.qualify_parent()
            receipt["scheduler"]["status_counts"] = {"running": 1}
            with self.assertRaisesRegex(ValueError, "Complete qualified Stage289"):
                spec.qualify_parent()


if __name__ == "__main__":
    unittest.main()
