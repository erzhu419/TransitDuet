import copy
import unittest

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch
from freq_hrl.experiments import pointmaze_actor_credit as scores
from freq_hrl.experiments import pointmaze_credit_reliability as reliability
from freq_hrl.experiments import pointmaze_independent_credit as independent
from freq_hrl.experiments import pointmaze_mc_control_variate as experiment
from tests import test_pointmaze_credit_reliability as fixtures
from scripts import pointmaze_mc_control_variate_stage70_spec as spec
from scripts.submit_pointmaze_mc_control_variate_stage70_scheduleurm import task_specification, qualification_task


class MCControlVariateTest(unittest.TestCase):
    def test_covariance_identity_and_variance_reduction(self):
        common = np.array([[8., 1.], [8., -1.], [-8., 1.], [-8., -1.]] * 2)
        state = common.copy()
        state[:, 0] = 0.
        row = experiment.variance_decomposition(common, state)
        self.assertAlmostEqual(row["state_over_common_variance"], 1 / 65)
        self.assertAlmostEqual(row["trace_common_baseline_covariance"], 64 * 8 / 7)
        self.assertLess(row["identity_relative_error"], 1e-15)
        inflated = experiment.variance_decomposition(common, common * 2)
        self.assertAlmostEqual(inflated["state_over_common_variance"], 4.)
        self.assertLess(inflated["variance_reduction_fraction"], 0.)

    def test_fixed_state_baseline_cancels_in_gaussian_expected_score(self):
        torch.set_num_threads(1)
        actor = GaussianActor(1, 1, 0, 0.)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        before = copy.deepcopy(actor.state_dict())
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        action = np.tile(np.sqrt(2.) * nodes, 2).astype(np.float32).reshape(-1, 1)
        state = np.repeat([0., 1.], 16).astype(np.float32).reshape(-1, 1)
        baseline = 5. + 8. * state[:, 0]
        quadrature = np.tile(weights / np.sqrt(np.pi) / 2., 2)
        with torch.no_grad():
            logp, _ = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))
        lower = LevelTrajectoryBatch(state=state, action=action, reward=np.zeros(32, dtype=np.float32),
            duration=np.ones(32, dtype=np.int64), done=np.ones(32, dtype=np.float32),
            old_logp=logp.numpy(), old_value=np.zeros(32, dtype=np.float32))
        reward = action[:, 0] + baseline
        g, _, _ = scores.actor_gradients(actor, lower,
            {"common": reward * quadrature * 32, "state": (reward - baseline) * quadrature * 32},
            clip_ratio=.2, chunk_size=64)
        np.testing.assert_allclose(g["common"], g["state"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_shared_score_pass_reproduces_stage69_and_keeps_raw_normalized_distinct(self):
        actor, lower, signal = fixtures.CreditReliabilityTest().data()
        signals = {"mc_common": signal["mc"], "mc_control": signal["mc"] - 4.,
            "mc_factored": signal["mc"] - np.arange(12), "gae_control": signal["gae"], "gae_factored": signal["gae"] * 2.}
        g, arrays, mask, cost = reliability.episode_scores(actor, lower, signals, horizon=3, clip_ratio=.2)
        self.assertEqual(cost["actor_score_backward_batches"], 7 * cost["actor_score_forward_batches"])
        batches, old_batches = [], []
        for idx in ((0, 1), (2, 3)):
            directions = reliability.fold_gradients(g, arrays, idx)
            gradients = {k: v[list(idx)] for k, v in g.items()}
            batches.append({"gradients": gradients, "directions": directions})
            old = {"gradients": {"mc_common": gradients["mc_common"], "mc_normalized": gradients["gae_control"],
                "mc_factored": gradients["gae_factored"]}, "directions": {"mc_normalized": directions["gae_control"],
                "mc_factored": directions["gae_factored"]}}
            for t in ("mc_normalized", "mc_factored"):
                old["directions"][t + "_entropy"] = old["directions"][t] + .01 * directions["entropy"]
            old_batches.append(old)
        observed = experiment.compare_batches(batches, mask)
        source = independent.compare_batches(old_batches[0], old_batches, mask)
        experiment.reproduce_stage69(observed, source)
        self.assertEqual(observed["mean"]["within_raw"]["mc_factored"]["count"], 1)
        self.assertEqual(observed["mean"]["noise"]["mc_factored"]["episodes"], 4)
        self.assertNotEqual(observed["mean"]["within_raw"]["mc_control"]["mean"],
            observed["mean"]["within_normalized"]["mc_control"]["mean"])
        source["mean"]["raw_episode_noise"]["mc_common"]["covariance_trace"] *= 2
        with self.assertRaises(AssertionError):
            experiment.reproduce_stage69(observed, source)

    def test_fixed_archives_cost_and_scheduler_roster(self):
        budget = spec.budget(preflight=False)
        self.assertEqual(budget["archive_episodes"] * 8, 1024)
        self.assertEqual(budget["reconstructed_lower_calls"] * 8, 1228800)
        self.assertEqual(budget["actor_score_forward_batches"] * 8, 2048)
        self.assertEqual(budget["actor_score_backward_batches"] * 8, 14336)
        for preflight in (False, True):
            for root in spec.roots(preflight=preflight):
                self.assertEqual(spec.seed_roles(root, preflight=preflight)["archive_batches"],
                    spec.source.seed_roles(root, preflight=preflight)["fresh_batches"])
                task = task_specification("unit_stage70", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (5, 4096))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage70", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))


if __name__ == "__main__":
    unittest.main()
