import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad

from freq_hrl.domains.transit import native_balanced_critic as learning
from scripts import run_native_transit_balanced_critic_stage247 as spec
from scripts.submit_native_transit_balanced_critic_stage247_scheduleurm import task_specification
from tests import test_native_transit_pair_critic_stage246 as fixture_module


class BalancedCriticTest(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.PairCriticTest()
        fixture.setUp()
        self.rows, self.data, self.draws = fixture.rows, fixture.data, fixture.draws
        self.cached, self.fit = learning.previous.fit_paths(self.rows, self.data, self.draws, fixture.cached, fixture.fit)

    def test_normalization_preserves_ratios_and_zero_or_single_pair_states(self):
        rng = np.random.default_rng(247)
        precision = np.exp(rng.normal(size=(4, 15, 2)))
        gate = np.array([[True]*6, [True, True, False, False, False, False],
            [True, False, False, False, False, False], [False]*6])
        mask = gate[:, learning.previous.LO] & gate[:, learning.previous.HI]
        for per_head in (False, True):
            balanced, ledger = learning.balanced_precision(precision, gate, per_head=per_head)
            self.assertEqual(ledger["active_states"], 2)
            self.assertLessEqual(ledger["mass_max_abs_difference"], 1e-12)
            np.testing.assert_array_equal(balanced[2:], 0.)
            if per_head:
                np.testing.assert_allclose(balanced[1, mask[1]], 1.)
                np.testing.assert_allclose((balanced/precision)[0], np.broadcast_to((balanced/precision)[0, :1], (15, 2)))
            else:
                ratios = (balanced/precision)[0]
                np.testing.assert_allclose(ratios, np.full((15, 2), ratios[0, 0]))
        first = learning.balanced_precision(precision, gate)[0]
        factors = np.array([.1, 7., 3., 4.])[:, None, None]
        np.testing.assert_allclose(first, learning.balanced_precision(precision*factors, gate)[0])

    def test_balanced_gradient_hessian_and_ineligible_precision_invariance(self):
        rng = np.random.default_rng(12)
        z, target, difference = rng.normal(size=(3, 6, 3)), rng.normal(size=(3, 6, 2)), rng.normal(size=3)
        precision = np.exp(rng.normal(size=(3, 15, 2)))
        gate = np.array([[True, True, True, False, False, False]]*3)
        theta = rng.normal(size=3)
        for per_head in (False, True):
            weights = learning.balanced_precision(precision, gate, per_head=per_head)[0]
            loss, hessian = learning.previous.pair_loss(z, target, weights, difference, gate)
            self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], theta), 1e-6)
            for column in range(3):
                step = np.eye(3)[column]*1e-5
                np.testing.assert_allclose((loss(theta+step)[1]-loss(theta-step)[1])/2e-5, hessian(theta)[:, column], atol=1e-8)
            other = precision.copy()
            other[:, ~(gate[0, learning.previous.LO] & gate[0, learning.previous.HI])] *= 100
            altered = learning.balanced_precision(other, gate, per_head=per_head)[0]
            mask = gate[:, learning.previous.LO] & gate[:, learning.previous.HI]
            np.testing.assert_array_equal(weights[mask], altered[mask])

    def test_frozen_controls_capacity_budget_and_unused_future_invariance(self):
        before = copy.deepcopy(self.cached)
        with patch.object(learning.joint, "fit_reference", side_effect=AssertionError("Raw retraining")):
            models, fit = learning.fit_paths(self.rows, self.data, self.draws, self.cached, self.fit)
        spec.qualify_fit(fit)
        self.assertEqual(self.cached, before)
        for o in learning.previous.OBJECTIVES: self.assertEqual(models[o], before[o])
        radii = {f["level"]: f["radius"] for f in self.fit["fits"]}
        for f in fit["fits"]:
            self.assertEqual(f["radius"], radii[f["level"]])
            self.assertLessEqual(f["norm"], f["radius"]+1e-10)
        for o in learning.NEW_OBJECTIVES:
            for level in learning.strength.lowrank.LEVELS:
                a, b = np.asarray(models[o]["weights"][level]), np.asarray(before["all_mse"]["weights"][level])
                np.testing.assert_allclose(a[0]-a[1], b[0]-b[1], atol=1e-12)
        unused = np.concatenate((self.draws, np.full_like(self.draws, 999.)), axis=1)
        self.assertEqual((models, fit), learning.fit_paths(self.rows, self.data, unused[:, :4], self.cached, self.fit))

    def test_old_selection_metrics_and_cost_accounting_reproduce(self):
        models, _ = learning.fit_paths(self.rows, self.data, self.draws, self.cached, self.fit)
        paths = learning.candidates(self.data, models)
        curves = {o: {c: learning.base.curve(self.data, self.draws, p) for c, p in ps.items()} for o, ps in paths.items()}
        blocks = [{"held_group": [r, 0], "half": h, "scales": [.1, .1], "curves": curves} for r in (397, 401) for h in (0, 1)]
        new, _ = learning.select(blocks)
        old, _ = learning.previous.select(blocks)
        for o in learning.previous.OBJECTIVES: self.assertEqual(new[o], old[o])
        pred = learning.predict(paths, new)
        stats = learning.metrics(self.data, self.draws, pred)
        control = learning.metrics(self.data, self.draws, {v: p for v, p in pred.items() if v in learning.previous.VARIANTS})
        current, expected = learning.aggregate([stats]), learning.previous.aggregate([control])
        self.assertEqual(learning.compare_cached({"held60": {"fresh8": current}}, {"held60": {"fresh8": expected}})["max_abs_difference"], 0.)
        for p in stats["all"]["variants"].values(): self.assertAlmostEqual(p["delta_cost"], p["delta_service"]+p["delta_risk"], places=12)

    def test_contract_and_unpinned_linux_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c["primary"], "state_gls_pooled_safe")
        self.assertEqual(c["budget"]["new_raw_reference_fits"], 0)
        self.assertEqual(c["budget"]["new_native_episodes"], 0)
        for key in (spec.PREFLIGHT, None):
            t = task_specification("unit_balanced", key)
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(any("results" in str(p) for p in t["stage_input_paths"]))


if __name__ == "__main__": unittest.main()
