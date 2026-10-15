import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_disjoint_critic as learning
from freq_hrl.domains.transit import native_frozen_future as sampling
from scripts import run_native_transit_disjoint_critic_stage244 as spec
from scripts.submit_native_transit_disjoint_critic_stage244_scheduleurm import task_specification
from tests.test_native_transit_variance_critic_stage243 import fixture


class DisjointCriticTest(unittest.TestCase):
    def setUp(self):
        self.rows, self.data, draws, self.scalar = fixture()
        self.a, self.b = draws[:, :4], draws[:, 4:]
        self.cached = learning.base.fit_paths(self.rows, self.data, self.a, self.scalar)[0]

    def test_matched_ols_controls_rms_budget_and_no_raw_refit(self):
        saved = copy.deepcopy(self.cached)
        with patch.object(learning.joint, "fit_reference", side_effect=AssertionError("Raw refit")):
            models, fit = learning.fit_paths(self.rows, self.data, self.a, self.b, self.cached)
        spec.qualify_fit(fit)
        self.assertEqual(self.cached, saved)
        self.assertIs(models["same"], self.cached)
        paths = learning.candidates(self.data, models)
        for key in learning.strength.CANDIDATES:
            for first, second in (("same", "calibration_only"), ("target_only", "disjoint")):
                for arm in ("combined", "split"):
                    np.testing.assert_array_equal(paths[first][arm][key]["cost"], paths[second][arm][key]["cost"])
                    np.testing.assert_array_equal(paths[first][arm][key]["chosen"], paths[second][arm][key]["chosen"])
            for m in learning.MODES:
                np.testing.assert_array_equal(paths[m]["combined"][key]["chosen"], paths[m]["split"][key]["chosen"])
        for fit in fit["fits"]:
            self.assertLessEqual(fit["norm"], fit["radius"]+1e-10)

    def test_factorial_changes_calibration_but_not_targets_of_ols(self):
        first, fit = learning.fit_paths(self.rows, self.data, self.a, self.b, self.cached)
        b = self.b.copy()
        b[:, :, 1:, 2] += .1
        b[..., 0] = b[..., 1]+b[..., 2]
        second, other = learning.fit_paths(self.rows, self.data, self.a, b, self.cached)
        for arm in ("combined", "split"):
            self.assertEqual(first["calibration_only"][arm], second["calibration_only"][arm])
        self.assertNotEqual(first["calibration_only"]["group"], second["calibration_only"]["group"])
        self.assertNotEqual(first["disjoint"]["combined"], second["disjoint"]["combined"])
        self.assertEqual(fit["noise"]["A"], other["noise"]["A"])
        self.assertNotEqual(fit["noise"]["B"], other["noise"]["B"])
        with self.assertRaisesRegex(ValueError, "counts must match"):
            learning.fit_paths(self.rows, self.data, self.a, self.b[:, :2], self.cached)

    def test_same_future_perfect_fit_hides_signal_and_zero_budget_is_reported(self):
        data = copy.deepcopy(self.data)
        data["components"] = self.a.mean(axis=1)[..., (2, 1)]
        data["heads"][..., 0] = data["components"].sum(axis=-1)
        values = self.a.mean(axis=1)[..., (0, 3)]
        scalar = learning.strength.lowrank.fit_path(self.rows, data, values, "original")[0]
        # Exact zero budget, not a floating-point near-zero residual.
        for w in scalar["weights"].values(): w[0] = [0.]*198
        cached = learning.base.fit_paths(self.rows, data, self.a, scalar)[0]
        models, fit = learning.fit_paths(self.rows, data, self.a, np.zeros_like(self.b), cached)
        self.assertTrue(all(g["prior_signal_energy"] == [0., 0.] for g in fit["noise"]["A"]["groups"]))
        self.assertTrue(all(g["prior_signal_energy"][1] > 0 for g in fit["noise"]["B"]["groups"]))
        self.assertTrue(any(f["OLS_budget_clipped"] for f in fit["fits"] if f["mode"] == "disjoint"))
        for arm in models["disjoint"].values():
            for weights in arm["weights"].values(): np.testing.assert_allclose(np.sum(weights, axis=0), 0., atol=1e-12)

    def test_validation_and_extra_futures_do_not_enter_fit_and_future_indices_disjoint(self):
        a, b, val = spec.draw_assignment((0, 2, 4, 6), (1, 3, 5, 7), 1)
        self.assertEqual((a, b, val), ([4, 6], [5, 7], [0, 2]))
        self.assertFalse(set(a) & set(b) or set(b) & set(val) or set(a) & set(val))
        first = learning.fit_paths(self.rows, self.data, self.a, self.b, self.cached)
        unused = np.concatenate((self.a, self.b, np.full_like(self.a, 1000.)), axis=1)
        self.assertEqual(first, learning.fit_paths(self.rows, self.data, unused[:, :4], unused[:, 4:8], self.cached))
        with self.assertRaisesRegex(ValueError, "disjoint matched"):
            spec.draw_assignment((0, 1, 2, 3), (0, 4, 5, 6))

    def test_feasible_mse_and_regret_ignore_ineligible_truth(self):
        models, _ = learning.fit_paths(self.rows, self.data, self.a, self.b, self.cached)
        data = copy.deepcopy(self.data)
        data["originals"][:] = 0
        data["raw_logits"][:, 4:] = 1.
        data["raw_choices"] = learning.joint.binding.policy.selected(
            learning.joint.relative(data["heads"], data["originals"])[..., 0], data["raw_logits"], data["originals"])
        paths = learning.candidates(data, models)
        selection = {m: {a: {f: "3@1" for f in learning.base.FILTERS} for a in learning.base.ARMS} for m in learning.MODES}
        pred = learning.predict(paths, selection)
        first = learning.metrics(data, self.b, pred)
        changed = self.b.copy()
        changed[:, :, 4:, 0] += 100.
        second = learning.metrics(data, changed, pred)
        for v in learning.VARIANTS:
            for field in ("feasible_cost_mse", "gate_cost_regret"):
                self.assertEqual(first["all"]["variants"][v][field], second["all"]["variants"][v][field])
        self.assertEqual(learning.aggregate([first, first])["all"]["variants"], first["all"]["variants"])

    def test_conditional_mc_keeps_shared_draws_and_accepts_new_variants(self):
        key = (397, 1, 8)
        values = np.zeros((8, 6, 4))
        values[:, 1, 0] = np.arange(8)
        panel = {key: {"values": values}}
        cell = {"blocks": [{"states": [dict(zip(("root", "scene_seed", "probe_index"), key))], "choices": {"raw": [0], "new": [1]}}]}
        a = sampling.conditional_mc([cell], panel, variants=("raw", "new"))
        b = sampling.conditional_mc([cell, cell], panel, variants=("raw", "new"))
        self.assertEqual(a["variants"], b["variants"])
        self.assertEqual(a["variants"]["new"]["fields"]["cost"]["mean"], 3.5)

    def test_contract_and_scheduler_capacity(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c["budget"]["new_raw_reference_fits"], 0)
        self.assertEqual(c["budget"]["new_native_episodes"], 0)
        for key in (spec.PREFLIGHT, None):
            task = task_specification("unit_disjoint", key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(any("results" in str(p) for p in task["stage_input_paths"]))


if __name__ == "__main__": unittest.main()
