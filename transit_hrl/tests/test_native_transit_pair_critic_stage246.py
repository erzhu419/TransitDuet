import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad

from freq_hrl.domains.transit import native_pair_critic as learning
from scripts import run_native_transit_pair_critic_stage246 as spec
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import task_specification, require_completed_attempts
from tests import test_native_transit_feasible_critic_stage245 as fixture_module


class PairCriticTest(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.FeasibleCriticTest()
        fixture.setUp()
        self.rows, self.data, self.draws = fixture.rows, fixture.data, fixture.draws
        self.cached, self.fit = learning.previous.fit_paths(self.rows, self.data, self.draws, fixture.cached)

    def test_small_loss_scale_preserves_dual_accuracy(self):
        points = []
        for scale in (1., 1e-12):
            matrix = scale*np.array([[3., .4], [.4, 1.]])
            def loss(t):
                d = t-np.array([2., -1.])
                return float(d@matrix@d/2), matrix@d
            theta, fit = learning.optimizer.quadratic_fit(loss, lambda t: matrix, np.array([.2, .1]), radius=.5)
            self.assertLessEqual(fit["optimality"], 1e-8)
            points.append(theta)
        np.testing.assert_allclose(*points, atol=1e-12)

    def test_interior_optimum_does_not_infer_multiplier_from_roundoff(self):
        target = np.array([1e-5])
        def loss(t):
            d = t-target
            return float(1e-12+d@d/2), d
        theta = np.nextafter(target, np.zeros_like(target))
        _, fit = learning.optimizer.finish_fit(loss, np.zeros(1), theta, 0, radius=.5)
        self.assertLess(fit["optimality"], 1e-8)
        self.assertLess(fit["norm"], .001*fit["radius"])

    def test_recovery_gate_keeps_failed_attempts_and_requires_successful_leaf(self):
        records = {"old": {"id": "old", "signature": "cell", "status": "failed", "exit_code": 1},
            "cancel": {"id": "cancel", "signature": "cell", "status": "cancelled", "parent_id": "old"},
            "fixed": {"id": "fixed", "signature": "cell", "status": "done", "exit_code": 0, "parent_id": "cancel"}}
        before = copy.deepcopy(records)
        require_completed_attempts(list(records.values()), [{"signature": "cell"}], records)
        self.assertEqual(records, before)
        with self.assertRaises(SystemExit): require_completed_attempts([records["old"], records["cancel"]], [{"signature": "cell"}], records)

    def test_pair_gradient_hessian_and_ineligible_labels(self):
        rng = np.random.default_rng(246)
        z, target, difference = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 6, 2)), rng.normal(size=3)
        precision = np.exp(rng.normal(size=(4, 15, 2)))
        gate = np.array([[True, True, True, False, False, False]]*4)
        theta = rng.normal(size=3)
        loss, hessian = learning.pair_loss(z, target, precision, difference, gate)
        self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], theta), 1e-6)
        for column in range(3):
            step = np.eye(3)[column]*1e-5
            np.testing.assert_allclose((loss(theta+step)[1]-loss(theta-step)[1])/2e-5, hessian(theta)[:, column], atol=1e-8)
        other = target.copy()
        other[:, 3:] += 100
        poisoned, _ = learning.pair_loss(z, other, precision, difference, gate)
        self.assertEqual(loss(theta)[0], poisoned(theta)[0])
        np.testing.assert_array_equal(loss(theta)[1], poisoned(theta)[1])
        empty, _ = learning.pair_loss(z, target, precision, difference, np.zeros_like(gate))
        self.assertEqual(empty(theta)[0], 0)
        np.testing.assert_array_equal(empty(theta)[1], np.zeros_like(theta))

    def test_paired_variance_retains_covariance_and_zero_head_is_not_tail_certificate(self):
        values = self.draws.copy()
        first, ledger = learning.pair_precision(self.rows, values, .1)
        shared = np.arange(4)[None, :, None]*.1
        values[..., 1] += shared
        values[..., 2] += shared
        second, changed = learning.pair_precision(self.rows, values, .1)
        np.testing.assert_allclose(first, second, atol=1e-12)
        np.testing.assert_allclose(ledger["pooled_variance_of_mean"], changed["pooled_variance_of_mean"], atol=1e-12)
        values[..., 1] = 0
        precision, ledger = learning.pair_precision(self.rows, values, .1)
        np.testing.assert_array_equal(precision[..., 1], np.ones((18, 15)))
        self.assertEqual(ledger["zero_observed_variance_heads"], ["risk"])
        self.assertTrue(np.isfinite(precision).all())
        np.testing.assert_allclose(precision.mean(axis=(0, 1)), 1.)

    def test_frozen_controls_capacity_budget_and_head_difference(self):
        before = copy.deepcopy(self.cached)
        with patch.object(learning.joint, "fit_reference", side_effect=AssertionError("Raw retraining")):
            models, fit = learning.fit_paths(self.rows, self.data, self.draws, self.cached, self.fit)
        spec.qualify_fit(fit)
        self.assertEqual(self.cached, before)
        for objective in learning.previous.OBJECTIVES: self.assertEqual(models[objective], before[objective])
        for o in learning.NEW_OBJECTIVES:
            for level in learning.strength.lowrank.LEVELS:
                a, b = np.asarray(models[o]["weights"][level]), np.asarray(before["all_mse"]["weights"][level])
                np.testing.assert_allclose(a[0]-a[1], b[0]-b[1], atol=1e-12)
        radii = {f["level"]: f["radius"] for f in self.fit["fits"]}
        for f in fit["fits"]:
            self.assertEqual(f["radius"], radii[f["level"]])
            self.assertLessEqual(f["norm"], f["radius"]+1e-10)
            self.assertLessEqual(f["loss_after"], f["loss_before"]+1e-10)
        unused = np.concatenate((self.draws, np.full_like(self.draws, 999.)), axis=1)
        self.assertEqual((models, fit), learning.fit_paths(self.rows, self.data, unused[:, :4], self.cached, self.fit))

    def test_pair_metrics_accounting_and_cached_control_reproduction(self):
        models, _ = learning.fit_paths(self.rows, self.data, self.draws, self.cached, self.fit)
        selection = {o: {f: "3@1" for f in learning.FILTERS} for o in learning.OBJECTIVES}
        paths = learning.candidates(self.data, models)
        pred = learning.predict(paths, selection)
        stats = learning.metrics(self.data, self.draws, pred)
        old = learning.previous.previous.metrics(self.data, self.draws, {v: p for v, p in pred.items() if v in learning.previous.VARIANTS})
        current, expected = learning.aggregate([stats]), learning.previous.aggregate([old])
        self.assertEqual(learning.compare_cached({"held60": {"fresh8": current}}, {"held60": {"fresh8": expected}})["max_abs_difference"], 0.)
        for piece in (stats["all"], *stats["by_root"].values()):
            for p in piece["variants"].values():
                self.assertAlmostEqual(p["delta_cost"], p["delta_service"]+p["delta_risk"], places=12)
                self.assertLessEqual(p["changed_harm_rate"], p["changed_rate"])
                self.assertTrue(0 <= p["feasible_pair_sign_error"] <= 1)

    def test_extended_selection_preserves_old_objectives(self):
        models, _ = learning.fit_paths(self.rows, self.data, self.draws, self.cached, self.fit)
        paths = learning.candidates(self.data, models)
        curves = {o: {c: learning.base.curve(self.data, self.draws, p) for c, p in ps.items()} for o, ps in paths.items()}
        blocks = [{"held_group": [r, 0], "half": h, "scales": [.1, .1], "curves": curves} for r in (397, 401) for h in (0, 1)]
        new, _ = learning.select(blocks)
        old, _ = learning.previous.select(blocks)
        for o in learning.previous.OBJECTIVES: self.assertEqual(new[o], old[o])

    def test_contract_and_unpinned_linux_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c["primary"], "pair_gls_pooled_safe")
        self.assertEqual(c["budget"]["new_raw_reference_fits"], 0)
        self.assertEqual(c["budget"]["new_native_episodes"], 0)
        for key in (spec.PREFLIGHT, None):
            task = task_specification("unit_pair", key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(any("results" in str(p) for p in task["stage_input_paths"]))


if __name__ == "__main__": unittest.main()
