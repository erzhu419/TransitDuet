from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_zero_abstention_stage291 as spec
from scripts import submit_native_transit_zero_abstention_stage291_scheduleurm as submit


def rows():
    native = spec.source.base.source
    return [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p}
        for k in native.keys("train") for p in native.PROBES]


class ZeroAbstentionTest(unittest.TestCase):
    def test_zero_events_not_zero_mean_and_no_reselection(self):
        cube = np.arange(6).reshape(1, 6, 1)
        screening = np.zeros((6, 8, 6, 4))
        screening[1, :, 1, 1] = -1
        screening[2, :4, 1, 1] = 1
        screening[2, 4:, 1, 1] = -1
        screening[3, :, 1, 1] = 1
        screening[:, :, 2, 1] = -5
        raw = np.zeros((1, 6), int)
        proposal = np.array([[1, 1, 1, 1, 0, 1]])
        independent, _, _ = spec.source.screen(screening, proposal, raw)
        legacy = np.stack((raw, raw, proposal, proposal, independent, proposal), axis=1)
        choices, probability, eligible, keep, counts = spec.policy(legacy, screening, cube)
        np.testing.assert_array_equal(choices[:, 3], [[0, 1, 1, 0, 0, 0]])
        np.testing.assert_allclose(probability, [[.5, .5, .5, 0, 0, .5]])
        self.assertEqual(counts["kept_changes"], 2)
        self.assertEqual(counts["eligible_changes"], 4)
        self.assertEqual(counts["expected_thin_changes"], 2.)
        self.assertTrue(np.all(keep <= eligible))

    def test_exact_fixed_count_expectation_and_shared_profile_noise(self):
        cube = np.arange(3).reshape(1, 3, 1)
        eligible = np.ones((1, 3), bool)
        keep = np.array([[True, False, False]])
        probability, _ = spec.thin_probabilities(cube, eligible, keep)
        draws = np.zeros((3, 8, 6, 4))
        draws[:, :, 1, 0] = np.array([-2, 2, 2])[:, None]
        draws[:, :, 1, 1] = np.arange(8)[None]-3
        raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
        choices = np.stack((raw, raw, proposal, np.where(keep, proposal, raw), raw, proposal), axis=1)
        actual, _ = spec.effects(choices, probability, proposal, draws)
        enumerated = []
        for retained in combinations(range(3), 1):
            mask = np.zeros(3)
            mask[list(retained)] = 1
            enumerated.append((draws[:, :, 1, 0]*mask[:, None]).mean())
        self.assertAlmostEqual(actual[:, :, 4, 0].mean(), np.mean(enumerated))
        repeated, _ = spec.effects(np.repeat(choices, 16, axis=0), np.repeat(probability, 16, axis=0),
            np.repeat(proposal, 16, axis=0), draws)
        np.testing.assert_allclose(actual, repeated, atol=1e-15)

    def test_bootstrap_rematches_counts_not_frozen_probabilities(self):
        cube = np.arange(3).reshape(1, 3, 1)
        eligible, keep = np.ones((1, 3), bool), np.array([[True, False, False]])
        proposal_mean = np.zeros((1, 3, 4))
        proposal_mean[0, :, 0] = [-2, 2, 2]
        means = np.zeros((1, 3, 6, 4))
        result = spec.bootstrap_samples(means, proposal_mean, cube, eligible, keep, count=48)
        ix = np.random.default_rng(291).choice(3, (48, 3), replace=True)
        k = (ix == 0).sum(axis=1)
        expected = (proposal_mean[0, ix, 0].sum(axis=1)/3)*(k/3)
        np.testing.assert_allclose(result[:, 4, 0], expected)
        self.assertTrue(np.any(k == 0))
        np.testing.assert_array_equal(result[k == 0, 4, 0], 0.)

    def test_joint_and_information_conclusions_are_separate(self):
        result = {"comparisons": {"zero_abstain_vs_raw": {"prefix_CI95": {"cost": [-2, -1], "wait_min": [-2, -1]}},
            "zero_abstain_vs_matched_thin": {"prefix_CI95": {"cost": [-2, 1], "fleet": [-2, -1]}}},
            "halves": {h: {"zero_abstain": {"cost": -1, "wait_min": -1}} for h in ("first4", "last4")}}
        flags = spec.primary_flags(result)
        self.assertTrue(flags["joint_retention_supported"])
        self.assertTrue(flags["fleet_information_supported"])
        self.assertFalse(flags["cost_information_supported"])
        result["halves"]["last4"]["zero_abstain"]["cost"] = 1
        self.assertFalse(spec.primary_flags(result)["joint_retention_supported"])
        result["comparisons"]["zero_abstain_vs_raw"]["prefix_CI95"]["wait_min"] = [1, 2]
        self.assertFalse(spec.primary_flags(result)["primary_joint_CI_passed"])

    def test_train_cache_roundtrip_freezes_and_count_matching(self):
        rr = rows()
        cube = spec.row_cube(rr)
        good = np.array([r["replica"] < 2 for r in rr])
        values = np.zeros((360, 24, 6, 4))
        values[..., 0] = 2
        values[..., 2] = 2
        values[..., 3] = 1
        values[:, :, 0] = 0
        values[:, :8, 1] = [-2, -3, 1, -.5]
        values[:, 8:16, 1, 1] = np.where(good, -1, 0)[:, None]
        values[:, 8:, 1, 2] = -1
        values[:, 8:, 1, 3] = -.5
        values[:, 16:, 1, 1] = np.where(good, -1, 3)[:, None]
        values[:, 8:, 1, 0] = values[:, 8:, 1, 1]+values[:, 8:, 1, 2]
        data = {"raw_logits": -np.ones((360, 6)), "originals": np.zeros(360, int), "raw_choices": np.zeros(360, int)}
        cached = np.zeros((16, 6, 2, 360), int)
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), patch.object(spec.source, "ROOT", Path(temp)):
            path = spec.directory("toy291")
            spec.write_json(path/"preregistration.json", {"contract": spec.contract()})
            for name, (p, s, _) in spec.QUERIES.items():
                legacy, _, _ = spec.source.compose([data]*16, values[:, spec.BANKS[p]], values[:, spec.BANKS[s]],
                    values[:, sorted(spec.BANKS[p]+spec.BANKS[s])], cached)
                spec.write_json(spec.source.directory(spec.SOURCE_RUN)/"report_raw"/(name+".json"),
                    {"arms": list(spec.source.ARMS), "query": spec.QUERIES[name], "choices": legacy.tolist()})
            calls = []

            def load(indices):
                calls.append(indices)
                if indices == spec.BANKS[1]:
                    self.assertTrue((path/"report_raw/primary_proposal.json").exists())
                if indices == spec.BANKS[2]:
                    ledger = spec.read(path/"report_raw/primary.json")
                    np.testing.assert_allclose(ledger["matched_thin_probabilities"], 1/3)
                    np.testing.assert_array_equal(np.asarray(ledger["choices"])[:, 3], np.broadcast_to(good, (16, 360)))
                return values[:, indices].copy()

            def load_rows(run, role):
                self.assertEqual((run, role), (spec.source.base.SOURCE_RUN, "train"))
                return rr

            kernel = spec.bootstrap_samples
            with patch.object(spec.source.base, "load_draws", side_effect=load), \
                    patch.object(spec.source.base.source, "load_rows", side_effect=load_rows), \
                    patch.object(spec, "bootstrap_samples", side_effect=lambda *args: kernel(*args, count=32)), \
                    patch.object(np, "load", side_effect=AssertionError("No model/prediction files needed")), \
                    patch.object(spec.source.base.source, "load_panels", side_effect=AssertionError("No held/full panels")), \
                    patch.object(spec.source.base.source, "fit", side_effect=AssertionError("No fit")), \
                    patch.object(spec.source.base.source, "collect", side_effect=AssertionError("No simulation")):
                spec.analyze("toy291")
            result = spec.read(path/"report/result.json")
            self.assertEqual(calls, list(spec.BANKS))
            self.assertEqual(len(result["queries"]), 6)
            self.assertTrue(result["joint_retention_supported"])
            self.assertTrue(result["cost_information_supported"])
            for q in result["queries"].values():
                self.assertAlmostEqual(q["matching"]["expected_thin_changes"], q["matching"]["kept_changes"])
            self.assertEqual(result["held_prefixes_read"], 0)
            self.assertEqual(result["new_fits"], 0)
            self.assertEqual(result["new_native_episodes"], 0)

    def test_scheduler_metadata_resources_and_report_only_pull(self):
        task = submit.task_specification("toy291")
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 3072, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn(str(spec.directory("toy291")), task["stage_input_paths"])
        self.assertEqual(task["result_dir"], str(spec.directory("toy291")/"report"))

    def test_parent_gate_requires_completed_stage290(self):
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}},
            "stage289_control_replay_exact": True, "held_prefixes_read": 0}

        def read(path):
            return receipt if path.name == "completion_receipt.json" else {"contract": spec.source.contract()}

        with patch.object(spec, "read", side_effect=read):
            spec.qualify_parent()
            receipt["scheduler"]["status_counts"] = {"running": 1}
            with self.assertRaisesRegex(ValueError, "Complete qualified Stage290"):
                spec.qualify_parent()


if __name__ == "__main__":
    unittest.main()
