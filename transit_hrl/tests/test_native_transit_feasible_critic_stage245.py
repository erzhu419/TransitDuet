import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad

from freq_hrl.domains.transit import native_feasible_critic as learning
from scripts import run_native_transit_feasible_critic_stage245 as spec
from scripts.submit_native_transit_feasible_critic_stage245_scheduleurm import task_specification
from tests.test_native_transit_variance_critic_stage243 import fixture


class FeasibleCriticTest(unittest.TestCase):
    def setUp(self):
        self.rows, self.data, draws, scalar = fixture()
        self.draws = draws[:, 4:]
        self.data["heads"] = learning.joint.relative(self.data["heads"], self.data["originals"])
        self.data["components"] = learning.joint.relative(self.data["components"], self.data["originals"])
        self.data["raw_choices"] = learning.joint.binding.policy.selected(
            self.data["heads"][..., 0], self.data["raw_logits"], self.data["originals"])
        cached = learning.base.fit_paths(self.rows, self.data, draws[:, :4], scalar)[0]
        self.cached = learning.previous.fit_paths(self.rows, self.data, draws[:, :4], self.draws, cached)[0]

    def test_explicit_radius_does_not_shrink_to_initial_update(self):
        def loss(t): return float(np.sum((t-2)**2)/2), t-2
        initial = np.array([.1])
        default, _ = learning.optimizer.quadratic_fit(loss, lambda t: np.eye(1), initial)
        explicit, fit = learning.optimizer.quadratic_fit(loss, lambda t: np.eye(1), initial, radius=1.)
        self.assertAlmostEqual(default[0], .1)
        self.assertAlmostEqual(explicit[0], 1.)
        self.assertEqual(fit["radius"], 1.)
        z = np.zeros((1, 6, 1))
        z[0, 1, 0] = 1.
        truth = np.array([[0., -2., 0., 0., 0., 0.]])
        gate = np.array([[True, True, False, False, False, False]])
        rank, hessian = learning.optimizer.losses(z, np.zeros((1, 6)), truth, gate, np.array([0]))["rank_regret"]
        theta, fit = learning.optimizer.constrained_fit(rank, hessian, np.zeros(1), radius=1.)
        self.assertAlmostEqual(theta[0], -1., places=4)
        self.assertEqual(fit["radius"], 1.)

    def test_component_gradient_hessian_and_ineligible_label_invariance(self):
        rng = np.random.default_rng(245)
        z, target = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 6, 2))
        precision, difference = np.exp(rng.normal(size=(4, 6, 2))), rng.normal(size=3)
        gate = np.array([[True, True, True, False, False, False]]*4)
        anchor, theta = np.zeros(4, dtype=int), rng.normal(size=3)
        loss, hessian = learning.component_mse(z, target, precision, difference, gate, anchor)
        self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], theta), 1e-6)
        for column in range(3):
            d = np.eye(3)[column]*1e-5
            np.testing.assert_allclose((loss(theta+d)[1]-loss(theta-d)[1])/2e-5, hessian(theta)[:, column], atol=1e-8)
        poisoned = target.copy()
        poisoned[:, 3:] += 100.
        other, _ = learning.component_mse(z, poisoned, precision, difference, gate, anchor)
        self.assertEqual(loss(theta)[0], other(theta)[0])
        np.testing.assert_array_equal(loss(theta)[1], other(theta)[1])

    def test_fixed_cached_control_capacity_difference_and_loss_improvement(self):
        before = copy.deepcopy(self.cached)
        with patch.object(learning.joint, "fit_reference", side_effect=AssertionError("Raw reference refit")):
            models, fit = learning.fit_paths(self.rows, self.data, self.draws, self.cached)
        spec.qualify_fit(fit)
        self.assertEqual(self.cached, before)
        self.assertEqual(models["all_mse"], self.cached["disjoint"]["variance_group"])
        for objective in learning.OBJECTIVES[1:]:
            for level in learning.strength.lowrank.LEVELS:
                a, b = np.array(models[objective]["weights"][level]), np.array(models["all_mse"]["weights"][level])
                np.testing.assert_allclose(a[0]-a[1], b[0]-b[1], atol=1e-12)
        for f in fit["fits"]:
            self.assertLessEqual(f["norm"], f["radius"]+1e-10)
            self.assertLessEqual(f["loss_after"], f["loss_before"]+1e-10)
        paths = learning.candidates(self.data, models)
        control = learning.base.candidates(self.data, self.cached["disjoint"]["variance_group"])
        for candidate in learning.strength.CANDIDATES:
            np.testing.assert_array_equal(paths["all_mse"][candidate]["cost"], control[candidate]["cost"])
            np.testing.assert_array_equal(paths["all_mse"][candidate]["chosen"], control[candidate]["chosen"])

    def test_all_feasible_mse_matches_cached_gls_with_frozen_contrast(self):
        models, _ = learning.fit_paths(self.rows, self.data, self.draws, self.cached)
        x = learning.joint.relative(self.data["x"][..., :198], self.data["originals"])
        for level in learning.strength.lowrank.LEVELS:
            first, second = np.array(models["all_mse"]["weights"][level]), np.array(models["feasible_mse"]["weights"][level])
            np.testing.assert_allclose(x@(first-second).T, 0., atol=1e-11)

    def test_unused_future_poisoning_does_not_enter_fit(self):
        first = learning.fit_paths(self.rows, self.data, self.draws, self.cached)
        bank = np.concatenate((self.draws, np.full_like(self.draws, 999.)), axis=1)
        self.assertEqual(first, learning.fit_paths(self.rows, self.data, bank[:, :4], self.cached))

    def test_contract_scheduler_and_root_is_diagnostic_only(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c["budget"]["new_raw_reference_fits"], 0)
        self.assertEqual(c["budget"]["new_native_episodes"], 0)
        for key in (spec.PREFLIGHT, None):
            t = task_specification("unit_feasible", key)
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(any("results" in str(p) for p in t["stage_input_paths"]))


if __name__ == "__main__": unittest.main()
