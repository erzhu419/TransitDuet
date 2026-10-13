import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_native_transit_best_action_stage218 as spec
from scripts.submit_native_transit_best_action_stage218_scheduleurm import task_specification


def fixture():
    rows = [{"root": root, "scenario": regime,
        "scene_seed": spec.prefixes.scene_seed(root, regime, replica), "probe_index": probe, "original_choice": 0}
        for root in spec.prefixes.ROOTS for regime in spec.prefixes.SCENARIOS
        for replica in range(4) for probe in spec.prefixes.PROBES]
    return rows, {"mean_states": copy.deepcopy(rows)}


def panels_for(rows, costs):
    panels = {}
    for row, cost in zip(rows, costs):
        values = np.zeros((8, 6, 4))
        values[:, :, 0] = cost
        values[:, 4:, 3] = 1.
        panels[spec.state_key(row)] = {"values": values}
    return panels


class BestActionTest(unittest.TestCase):
    def test_cohorts_are_equal_balanced_complete_and_label_independent(self):
        rows, registered = fixture()
        views = spec.matched_views(rows, registered)
        keys = lambda rs: {spec.state_key(r) for r in rs}
        for name, rs in views.items():
            self.assertEqual(len(rs), 120)
            groups = {(r["root"], r["scene_seed"]) for r in rs}
            self.assertEqual(len(groups), 20)
            for group in groups:
                self.assertEqual(sum((r["root"], r["scene_seed"]) == group for r in rs), 6)
            for regime in spec.prefixes.SCENARIOS:
                self.assertEqual(sum(r["scenario"] == regime for r in rs), 24)
            if name.startswith("mixed"):
                self.assertEqual(sum(r["root"] == 397 for r in rs), 60)
        self.assertFalse(keys(views["mixed_a"]) & keys(views["mixed_b"]))
        self.assertEqual(keys(views["mixed_a"]) | keys(views["mixed_b"]), keys(rows))
        changed = [{**r, "labels": [{"actual_cost_delta": 1e9}]} for r in reversed(rows)]
        for name, rs in spec.matched_views(changed, registered).items():
            self.assertEqual(keys(rs), keys(views[name]))
        with self.assertRaisesRegex(ValueError, "original240"):
            spec.matched_views(rows[:-1], registered)

    def test_best_action_can_be_linear_when_nonwinning_order_is_not(self):
        rows = fixture()[0][:2]
        panels = panels_for(rows, [[0., -3., -2., -1., 2., 3.], [0., -3., -1., -2., 2., 3.]])
        design = np.tile(np.asarray([0., -3., -2., -1., 2., 3.])[None, :, None], (2, 1, 1))
        best, eligible, targets, coverage = spec.best_constraints(rows, panels, design, "first_last")
        full = spec.source.stable_constraints(rows, panels, design, np.zeros((2, 6, 2)), eligible, "first_last")["cost_order"]
        self.assertEqual(coverage["nonidentity_best_states"], 2)
        self.assertEqual(coverage["strict_constraints"], 2)
        self.assertEqual([r["best_choice"] for r in targets], [1, 1])
        self.assertTrue(spec.source.linear_witness(*best)["feasible"])
        self.assertFalse(spec.source.linear_witness(*full)["feasible"])
        self.assertFalse(spec.source.linear_witness(*tuple(np.concatenate((b, f)) for b, f in zip(best, full)))["feasible"])

    def test_identity_ties_are_weak_and_unsafe_actions_are_not_cost_competitors(self):
        row = fixture()[0][0]
        panels = panels_for([row], [[0., 0., 0., 1., -10., -20.]])
        design = np.arange(6.)[None, :, None]
        problem, eligible, targets, coverage = spec.best_constraints([row], panels, design, "first_last")
        self.assertTrue(eligible[0])
        self.assertEqual(targets[0]["best_choice"], 0)
        self.assertEqual(targets[0]["safe_menu"], [0, 1, 2, 3])
        self.assertEqual(coverage["strict_constraints"], 0)
        np.testing.assert_array_equal(problem[1], [0., 0., 0.])
        self.assertTrue(spec.source.linear_witness(*problem)["feasible"])

    def test_half_disagreements_and_preceding_near_ties_are_reported(self):
        rows = fixture()[0][:2]
        panels = panels_for(rows, [[0., -2., -1., 1., 2., 3.], [0., -1e-13, 1., 2., 3., 4.]])
        panels[spec.state_key(rows[0])]["values"][4:, 2, 0] = -3.
        design = np.arange(6.)[None, :, None].repeat(2, axis=0)
        problem, eligible, targets, coverage = spec.best_constraints(rows, panels, design, "first_last")
        self.assertFalse(eligible.any())
        self.assertEqual(targets, [])
        self.assertEqual(coverage["half_best_disagreement_states"], 1)
        self.assertEqual(coverage["preceding_near_tie_states"], 1)
        self.assertEqual(problem[0].shape, (0, 1))
        other = spec.best_constraints(rows, panels, design, "even_odd")[3]
        self.assertEqual(other["half_best_disagreement_states"], 0)
        self.assertEqual(other["eligible_states"], 1)

    def test_normalizer_uses_causal_features_not_labels_or_root_metadata(self):
        rows = [{"original_choice": 2}, {"original_choice": 0}]
        raw = np.random.default_rng(218).normal(size=(2, 6, 198))
        raw[..., 5] = 7.
        with patch.object(spec.source.wait_source.neural, "batches", return_value=(raw, None)):
            cost, wait, normalizer = spec.feature_design(rows)
        self.assertEqual(cost.shape, (2, 6, 98))
        self.assertEqual(wait.shape, (2, 6, 199))
        np.testing.assert_array_equal(cost, wait[..., :98])
        np.testing.assert_array_equal(cost[[0, 1], [2, 0]], np.zeros((2, 98)))
        self.assertEqual(normalizer["scale"][5], 1.)
        np.testing.assert_allclose(normalizer["center"], raw.mean(axis=(0, 1)))

    def test_same_target_full_control_contains_best_constraints(self):
        rows = fixture()[0][:2]
        panels = panels_for(rows, [[0., -3., -2., -1., 2., 3.], [0., -3., -1., -2., 2., 3.]])
        raw = np.random.default_rng(220).normal(size=(2, 6, 198))
        captured = []
        saved = {}
        witness = spec.source.linear_witness
        def record(d, m):
            captured.append((d.copy(), m.copy()))
            return witness(d, m)
        with patch.object(spec.source.wait_source.neural, "batches", return_value=(raw, None)), patch.object(spec.source, "linear_witness", side_effect=record):
            result = spec.diagnose_view(rows, panels, lambda p, r: saved.update({p: r}))
        for offset, partition in zip((0, 4), spec.PARTITIONS):
            matched, best = captured[offset+1], captured[offset+2]
            for whole, subset in zip(matched, best):
                np.testing.assert_array_equal(whole[:len(subset)], subset)
            self.assertEqual(result["partitions"][partition]["target_coverage"]["eligible_states"], 2)
            self.assertIn("coefficients", saved[partition]["witnesses"]["best_action"])
            self.assertNotIn("coefficients", result["partitions"][partition]["bounds"]["best_action"])

    def test_scheduler_is_one_cpu_unbound_report_only_no_rollouts(self):
        task = task_specification("unit_best_action")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"], {"views": 4, "max_linear_programs": 32,
            "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0})


if __name__ == "__main__":
    unittest.main()
