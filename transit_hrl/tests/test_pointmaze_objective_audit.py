import copy
from dataclasses import replace
import unittest

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch
from freq_hrl.experiments import pointmaze_actor_credit as scores
from freq_hrl.experiments import pointmaze_credit_reliability as reliability
from freq_hrl.experiments import pointmaze_objective_audit as experiment
from tests import test_pointmaze_credit_reliability as fixtures
from scripts import pointmaze_objective_audit_stage72_spec as spec
from scripts.submit_pointmaze_objective_audit_stage72_scheduleurm import task_specification, qualification_task


class ObjectiveAuditTest(unittest.TestCase):
    def test_exact_gaussian_episode_gradients_distinguish_objectives(self):
        torch.set_num_threads(1)
        actor = GaussianActor(1, 1, 0, 0.)
        with torch.no_grad():
            for p in actor.parameters():p.zero_()
        before = copy.deepcopy(actor.state_dict())
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        a0, a1 = np.meshgrid(np.sqrt(2.) * nodes, np.sqrt(2.) * nodes, indexing="ij")
        action = np.stack((a0.ravel(), a1.ravel()), 1).astype(np.float32).reshape(-1, 1)
        reward = np.stack((a0.ravel(), 2 * a0.ravel() + 3 * a1.ravel()), 1).astype(np.float32).reshape(-1)
        state = np.ones((512, 1), dtype=np.float32)
        with torch.no_grad():logp, _ = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))
        lower = LevelTrajectoryBatch(state=state, action=action, reward=reward,
            duration=np.ones(512, dtype=np.int64), done=np.tile([0., 1.], 256).astype(np.float32),
            old_logp=logp.numpy(), old_value=np.zeros(512, dtype=np.float32),
            value_state=np.tile([[1.], [.5]], (256, 1)).astype(np.float32))
        signals, discounted = experiment.objective_signals(lower, horizon=2, gamma=.4,
            discounted_location=.3, native_location=2.7)
        quadrature = np.repeat((weights[:, None] * weights[None, :] / np.pi).reshape(-1), 2) * 256
        g, mask, _ = scores.actor_gradients(actor, lower, {k: v * quadrature for k, v in signals.items()}, clip_ratio=.2, chunk_size=1024)
        for k, expected in {"mc_native": -3., "mc_native_zero": -3., "mc_common": -2.4, "mc_discounted_objective": -1.5}.items():
            np.testing.assert_allclose(g[k][~mask], expected, atol=2e-5, rtol=0)
            np.testing.assert_allclose(g[k][mask], 0., atol=2e-5, rtol=0)
        np.testing.assert_allclose(discounted.reshape(-1, 2)[:, 0], reward.reshape(-1, 2) @ [1., .4], atol=1e-12, rtol=0)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_gamma_one_equivalence_and_true_episode_recursion(self):
        _, lower, _ = fixtures.CreditReliabilityTest().data()
        lower = replace(lower, reward=np.arange(12, dtype=np.float32), done=np.tile([0., 0., 1.], 4).astype(np.float32),
            value_state=np.tile([[1.], [2/3], [1/3]], (4, 1)).astype(np.float32))
        signals, mc = experiment.objective_signals(lower, horizon=3, gamma=1., discounted_location=2., native_location=2.)
        np.testing.assert_array_equal(signals["mc_native"], signals["mc_common"])
        np.testing.assert_array_equal(signals["mc_discounted_objective"], signals["mc_common"])
        np.testing.assert_array_equal(mc.reshape(-1, 3)[:, 0], [3., 12., 21., 30.])
        with self.assertRaises(AssertionError):
            experiment.objective_signals(replace(lower, done=np.zeros(12, dtype=np.float32)),
                horizon=3, gamma=1., discounted_location=2., native_location=2.)

    def test_all_estimators_raw_noise_and_independent_references(self):
        actor, lower, s = fixtures.CreditReliabilityTest().data()
        signals = {k: s["mc"] * (i + 1) + s["gae"] for i, k in enumerate(spec.ESTIMATORS)}
        g, arrays, mask, cost = reliability.episode_scores(actor, lower, signals, horizon=3, clip_ratio=.2)
        self.assertEqual(cost["actor_score_backward_batches"], 10 * cost["actor_score_forward_batches"])
        batches = [{"gradients": {k: v[idx] for k, v in g.items()},
            "directions": reliability.fold_gradients(g, arrays, idx)} for idx in ([0, 1], [2, 3])]
        row = experiment.compare_batches(batches, mask)["mean"]
        self.assertEqual(row["noise"]["mc_native"]["episodes"], 4)
        self.assertEqual(row["cross_raw_native_reference"]["gae_control"]["count"], 2)
        expected = scores.cosine(g["gae_control"][[0, 1]].mean(0)[~mask], g["mc_native"][[2, 3]].mean(0)[~mask])
        other = scores.cosine(g["gae_control"][[2, 3]].mean(0)[~mask], g["mc_native"][[0, 1]].mean(0)[~mask])
        self.assertAlmostEqual(row["cross_raw_native_reference"]["gae_control"]["mean"], (expected + other) / 2)

    def test_historical_only_frame_and_scheduler_budget(self):
        budget = spec.budget(preflight=False)
        self.assertEqual(budget["calibration_archive_episodes"] * 8, 256)
        self.assertEqual(budget["probe_archive_episodes"] * 8, 1024)
        self.assertEqual(budget["reconstructed_lower_calls"] * 8, 1536000)
        self.assertEqual(budget["reconstructed_upper_calls"] * 8, 23040)
        self.assertEqual(budget["actor_score_forward_batches"] * 8, 2048)
        self.assertEqual(budget["actor_score_backward_batches"] * 8, 20480)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertFalse(set(roles["first_calibration"]).intersection(s for b in roles["archive_batches"] for s in b))
                self.assertEqual(roles["archive_batches"], spec.source.seed_roles(root, preflight=preflight)["archive_batches"])
                task = task_specification("unit_stage72", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (5, 4096))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage72", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])


if __name__ == "__main__":
    unittest.main()
