import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.experiments.pointmaze_actor_credit import actor_gradients, cosine, gradient_summary
from scripts import pointmaze_actor_credit_stage66_spec as spec
from scripts.submit_pointmaze_actor_credit_stage66_scheduleurm import task_specification, qualification_task


class ActorCreditTest(unittest.TestCase):
    def test_gaussian_score_gradient_entropy_and_chunk_weighting(self):
        actor = GaussianActor(1, 1, 0, 0.)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        state, action = np.zeros((4, 1), dtype=np.float32), np.arange(4, dtype=np.float32).reshape(-1, 1)
        with torch.no_grad():
            old_logp, _ = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))
        lower = SimpleNamespace(state=state, action=action, old_logp=old_logp.numpy(), size=4)
        advantage = np.array([-1., 1., -2., 2.], dtype=np.float32)
        before = copy.deepcopy(actor.state_dict())
        expected_mu = -float((action[:, 0] * advantage).mean())
        expected_sigma = -float(((action[:, 0] ** 2 - 1.) * advantage).mean())
        for chunk in (1, 3, 4):
            gradients, mask, cost = actor_gradients(actor, lower, {"gae": advantage, "mc": -advantage}, clip_ratio=.2, chunk_size=chunk)
            np.testing.assert_allclose(gradients["gae"][mask], [expected_sigma], atol=1e-6)
            np.testing.assert_allclose(gradients["gae"][~mask], [0., expected_mu], atol=1e-6)
            np.testing.assert_allclose(gradients["entropy"][mask], [-1.], atol=1e-6)
            np.testing.assert_allclose(gradients["entropy"][~mask], [0., 0.], atol=1e-6)
            self.assertAlmostEqual(gradient_summary(gradients, mask, .001)["mean"]["credit_cosine"], -1.)
            self.assertEqual(cost["actor_score_backward_batches"], cost["actor_score_forward_batches"] * 3)
            torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertIsNone(cosine(np.zeros(2), np.ones(2)))

    def test_frozen_roster_costs_and_complete_marker_dependency(self):
        self.assertEqual(spec.roots(preflight=False), spec.source.roots(preflight=False))
        full = spec.budget(preflight=False)
        self.assertEqual(full["archive_episodes"] * 8, 256)
        self.assertEqual(full["reconstructed_lower_calls"] * 8, 307200)
        self.assertEqual(full["actor_score_forward_batches"] * 8, 640)
        self.assertEqual(full["actor_score_backward_batches"] * 8, 1920)
        for preflight in (False, True):
            qualification = qualification_task("unit_stage66", preflight=preflight)
            self.assertEqual(len(qualification["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(qualification["result_dir"])
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_stage66", root, preflight=preflight)
                self.assertEqual(task["cpu"], 3)
                self.assertEqual(task["ram_mb"], 3072)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))


if __name__ == "__main__":
    unittest.main()
