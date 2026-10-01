import copy
import unittest

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_calibrated_cv as experiment
from freq_hrl.experiments import pointmaze_mc_control_variate as previous
from freq_hrl.experiments import pointmaze_actor_credit as scores
from freq_hrl.experiments import pointmaze_credit_reliability as reliability
from tests import test_pointmaze_credit_reliability as fixtures
from scripts import pointmaze_calibrated_cv_stage71_spec as spec
from scripts.submit_pointmaze_calibrated_cv_stage71_scheduleurm import task_specification, qualification_task


class CalibratedCVTest(unittest.TestCase):
    def test_streaming_fit_matches_full_covariance_without_clipping(self):
        rng = np.random.default_rng(17)
        common = rng.normal(size=(13, 7))
        h = common * .25 + rng.normal(size=common.shape) * .1
        g = {"mc_common": common, "mc_control": common - h, "mc_factored": common + h}
        streamed, full = experiment.CovarianceFit(), experiment.CovarianceFit()
        for idx in (slice(0, 2), slice(2, 7), slice(7, 13)):
            streamed.update({k: v[idx] for k, v in g.items()})
        full.update(g)
        a, b = streamed.finish(), full.finish()
        for t in a:
            for k in ("alpha", "common_variance", "baseline_variance", "trace_common_baseline_covariance", "calibration_optimal_variance"):
                self.assertAlmostEqual(a[t][k], b[t][k])
            self.assertEqual(a[t]["episodes"], 13)
            self.assertLess(a[t]["calibration_optimal_variance"], a[t]["common_variance"])
        self.assertGreater(a["control"]["alpha"], 1.)
        self.assertLess(a["factored"]["alpha"], -1.)
        centered = h - h.mean(0)
        expected = np.sum((common - common.mean(0)) * centered) / np.sum(centered * centered)
        self.assertAlmostEqual(a["control"]["alpha"], expected)

    def test_exact_zero_baseline_variance(self):
        common = np.array([[1., 2.], [3., -2.], [-1., 4.]])
        fit = experiment.CovarianceFit()
        fit.update({"mc_common": common, "mc_control": common - 3., "mc_factored": common})
        for row in fit.finish().values():
            self.assertEqual(row["alpha"], 0.)
            self.assertEqual(row["common_variance"], row["calibration_optimal_variance"])

    def test_candidate_scores_match_linear_combination_and_frozen_models(self):
        actor, lower, s = fixtures.CreditReliabilityTest().data()
        before = copy.deepcopy(actor.state_dict())
        signals = {"mc_common": s["mc"], "mc_control": s["mc"] - 4.,
            "mc_factored": s["mc"] - np.arange(12), "gae_control": s["gae"], "gae_factored": s["gae"] * 2.}
        coefficients = {"control": {"alpha": -.25}, "factored": {"alpha": 2.5}}
        signals.update(experiment.candidate_signals(signals, coefficients))
        g, arrays, mask, cost = reliability.episode_scores(actor, lower, signals, horizon=3, clip_ratio=.2)
        self.assertEqual(cost["actor_score_backward_batches"], 9 * cost["actor_score_forward_batches"])
        for t, row in coefficients.items():
            expected = g["mc_common"] - row["alpha"] * (g["mc_common"] - g["mc_" + t])
            np.testing.assert_allclose(g["mc_calibrated_" + t], expected, atol=2e-6, rtol=1e-5)
        folds = reliability.fold_gradients(g, arrays, range(4))
        direct, _, _ = scores.actor_gradients(actor, lower, {k: (v.reshape(-1) - v.mean()) / (v.std() + 1e-8)
            for k, v in arrays.items()}, clip_ratio=.2, chunk_size=1024)
        for t in coefficients:
            np.testing.assert_allclose(folds["mc_calibrated_" + t], direct["mc_calibrated_" + t], atol=2e-6, rtol=1e-5)
        batches = [{"gradients": {k: v[idx] for k, v in g.items()},
            "directions": reliability.fold_gradients(g, arrays, idx)} for idx in ([0, 1], [2, 3])]
        old = previous.compare_batches(batches, mask)
        observed = experiment.compare_batches(batches, mask)
        experiment.reproduce_stage70(observed, old)
        self.assertEqual(observed["mean"]["noise"]["mc_calibrated_control"]["episodes"], 4)
        old["all"]["control_variates"]["mc_control"]["baseline_empirical_noise"]["covariance_trace"] *= 2
        with self.assertRaises(AssertionError):
            experiment.reproduce_stage70(observed, old)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_frozen_roles_counts_and_dynamic_scheduler(self):
        budget = spec.budget(preflight=False)
        self.assertEqual(budget["calibration_archive_episodes"] * 8, 4096)
        self.assertEqual(budget["probe_archive_episodes"] * 8, 1024)
        self.assertEqual(budget["reconstructed_lower_calls"] * 8, 6144000)
        self.assertEqual(budget["reconstructed_upper_calls"] * 8, 92160)
        self.assertEqual(budget["actor_score_forward_batches"] * 8, 10240)
        self.assertEqual(budget["actor_score_backward_batches"] * 8, 59392)
        self.assertEqual(budget["control_variate_coefficient_fits"] * 8, 64)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertEqual(roles["calibration"], spec.source.values_source.seed_roles(root, preflight=preflight)["calibration"])
                self.assertEqual(roles["archive_batches"], spec.source.seed_roles(root, preflight=preflight)["archive_batches"])
                self.assertFalse(set(roles["calibration"]).intersection(s for b in roles["archive_batches"] for s in b))
                task = task_specification("unit_stage71", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (5, 4096))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage71", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])


if __name__ == "__main__":
    unittest.main()
