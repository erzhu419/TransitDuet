import copy
import unittest

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch, FrequencySeparatedActorCriticPPO
from freq_hrl.experiments import pointmaze_actor_credit as scores
from freq_hrl.experiments import pointmaze_credit_reliability as experiment
from scripts import pointmaze_credit_reliability_stage68_spec as spec
from scripts.submit_pointmaze_credit_reliability_stage68_scheduleurm import task_specification, qualification_task


class CreditReliabilityTest(unittest.TestCase):
    def data(self):
        torch.set_num_threads(1)
        actor = GaussianActor(1, 1, 0, 0.)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        state = np.arange(12, dtype=np.float32).reshape(-1, 1) / 6
        action = (np.arange(12, dtype=np.float32).reshape(-1, 1) - 3) / 4
        with torch.no_grad():
            old_logp, _ = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))
        lower = LevelTrajectoryBatch(state=state, action=action, reward=np.zeros(12, dtype=np.float32),
            duration=np.ones(12, dtype=np.int64), done=np.tile([0., 0., 1.], 4),
            old_logp=old_logp.numpy(), old_value=np.zeros(12, dtype=np.float32))
        signal = {"gae": np.array([1, 4, 2, 3, 1, 2, -2, 5, 1, 6, 3, -1], dtype=np.float32),
            "mc": np.array([-1, 3, 4, -3, 1, 2, 1, 6, 2, -2, 3, 4], dtype=np.float64)}
        return actor, lower, signal

    def test_episode_algebra_matches_direct_separately_normalized_fold_gradients(self):
        actor, lower, signal = self.data()
        before = copy.deepcopy(actor.state_dict())
        gradients, arrays, mask, cost = experiment.episode_scores(actor, lower, signal, horizon=3, clip_ratio=.2)
        self.assertEqual(cost["actor_score_forward_batches"], 4)
        self.assertEqual(cost["actor_score_backward_batches"], 16)
        for indices in (range(4), (0, 1), (0, 2), (0, 3)):
            selected = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in indices])
            partial = LevelTrajectoryBatch(**{k: None if v is None else v[selected] for k, v in vars(lower).items()})
            direct, _, _ = scores.actor_gradients(actor, partial,
                {k: FrequencySeparatedActorCriticPPO._normalize(v[selected]) for k, v in signal.items()},
                clip_ratio=.2, chunk_size=4)
            reconstructed = experiment.fold_gradients(gradients, arrays, indices)
            for key in direct:
                np.testing.assert_allclose(reconstructed[key], direct[key], atol=2e-6, rtol=1e-5)
        comparisons, _, count = experiment.compare_gradients({t: gradients for t in spec.TREATMENTS},
            {t: arrays for t in spec.TREATMENTS}, mask)
        self.assertEqual(count, 3)
        self.assertEqual(comparisons["mean"]["within"]["mc_factored"]["gae"]["count"], 3)
        self.assertEqual(comparisons["mean"]["cross_reference"]["mc_factored_GAE/mc_normalized_MC"]["count"], 6)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_all_partitions_zero_gradient_and_nonlinear_clip_detection(self):
        partitions = list(experiment.balanced_partitions(8))
        self.assertEqual(len(partitions), 35)
        unique = set()
        for a, b in partitions:
            self.assertEqual(len(a), len(b))
            self.assertFalse(set(a) & set(b))
            self.assertEqual(set(a) | set(b), set(range(8)))
            key = frozenset((frozenset(a), frozenset(b)))
            self.assertNotIn(key, unique)
            unique.add(key)
        self.assertEqual(len(list(experiment.balanced_partitions(2))), 1)
        zero = experiment.cosine_statistics([None, None])
        self.assertEqual((zero["count"], zero["defined"], zero["undefined"]), (2, 0, 2))
        self.assertIsNone(zero["mean"])
        actor, lower, signal = self.data()
        lower.old_logp -= 1.
        with self.assertRaisesRegex(ValueError, "linear fold"):
            experiment.episode_scores(actor, lower, signal, horizon=3, clip_ratio=.2)

    def test_roster_costs_dynamic_placement_and_completion_dependency(self):
        full = spec.budget(preflight=False)
        self.assertEqual(full["archive_episodes"] * 8, 256)
        self.assertEqual(full["actor_score_forward_batches"] * 8, 1024)
        self.assertEqual(full["actor_score_backward_batches"] * 8, 4096)
        self.assertEqual(full["balanced_partitions"] * 8, 1120)
        self.assertEqual(spec.roots(preflight=False), spec.source.roots(preflight=False))
        for preflight in (False, True):
            qualification = qualification_task("unit_stage68", preflight=preflight)
            self.assertEqual(len(qualification["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(qualification["result_dir"])
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_stage68", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))


if __name__ == "__main__":
    unittest.main()
