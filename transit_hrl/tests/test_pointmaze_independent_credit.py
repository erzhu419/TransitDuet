import copy
from dataclasses import replace
import unittest

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from freq_hrl.experiments import pointmaze_independent_credit as experiment
from freq_hrl.experiments import pointmaze_credit_reliability as reliability
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from tests import test_pointmaze_credit_reliability as fixtures
from scripts import pointmaze_independent_credit_stage69_spec as spec
from scripts.submit_pointmaze_independent_credit_stage69_scheduleurm import task_specification, qualification_task


class IndependentCreditTest(unittest.TestCase):
    def test_common_baseline_depends_only_on_existing_clock_and_frozen_location(self):
        _, lower, _ = fixtures.CreditReliabilityTest().data()
        clock = np.tile([1., 2 / 3, 1 / 3], 4).astype(np.float32)
        lower = replace(lower, value_state=np.column_stack((lower.state, clock)))
        result = experiment.common_baseline(lower, horizon=3, gamma=.9, rate_location=2.)
        np.testing.assert_allclose(result, np.tile([5.42, 3.8, 2.], 4), atol=1e-6)
        changed = replace(lower, reward=lower.reward + 20., state=lower.state + 50.)
        np.testing.assert_array_equal(experiment.common_baseline(changed, horizon=3, gamma=.9, rate_location=2.), result)
        with self.assertRaises(AssertionError):
            experiment.common_baseline(replace(lower, value_state=lower.value_state[::-1]), horizon=3, gamma=.9, rate_location=2.)

    def test_noise_unbiased_signal_and_negative_estimates_are_not_clamped(self):
        row = experiment.gradient_noise([[1., 0.], [-1., 0.]])
        self.assertEqual(row["covariance_trace"], 2.)
        self.assertEqual(row["unbiased_signal_power"], -1.)
        self.assertEqual(row["debiased_mean_snr"], -1.)
        exact = experiment.gradient_noise([[3., 4.], [3., 4.]])
        self.assertEqual(exact["unbiased_signal_power"], 25.)
        self.assertIsNone(exact["debiased_mean_snr"])

    def test_TD_identity_true_terminals_and_shared_score_pass_leave_actor_frozen(self):
        actor, lower, old_signal = fixtures.CreditReliabilityTest().data()
        lower = replace(lower, reward=np.arange(12, dtype=np.float32) / 4)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=1, upper_action_dim=1,
            lower_state_dim=1, lower_action_dim=1, hidden_dim=4, gamma=.9, gae_lambda=.8))
        mc = calibration.monte_carlo_returns(lower, .9)
        pred = np.array([4., 5., -1., 100., -20., 8., 6., 2., -2., 10., 1., 3.], dtype=np.float32)
        advantage, _ = model._gae(lower.reward, lower.done, lower.duration, pred)
        row = experiment.td_attribution(lower, pred, mc, advantage, gamma=.9, lam=.8, horizon=3, period=2)
        self.assertLess(row["identity_max_abs_error"], 1e-5)
        self.assertEqual(row["renewal_TD"]["count"], 4)
        np.testing.assert_allclose(advantage[2::3], (mc - pred)[2::3], atol=1e-6)
        with self.assertRaises(AssertionError):
            experiment.td_attribution(replace(lower, done=np.zeros(12)), pred, mc, advantage,
                gamma=.9, lam=.8, horizon=3, period=2)
        before = copy.deepcopy(actor.state_dict())
        signals = {"mc_common": old_signal["mc"], "mc_normalized": old_signal["gae"], "mc_factored": old_signal["gae"] * 2.}
        gradients, arrays, mask, cost = reliability.episode_scores(actor, lower, signals, horizon=3, clip_ratio=.2)
        self.assertEqual(cost["actor_score_backward_batches"], 5 * cost["actor_score_forward_batches"])
        batches = []
        for idx in ((0, 1), (2, 3)):
            direction = reliability.fold_gradients(gradients, arrays, idx)
            for t in spec.TREATMENTS:
                direction[t + "_entropy"] = direction[t] + .01 * direction["entropy"]
            batches.append({"gradients": {k: v[list(idx)] for k, v in gradients.items()}, "directions": direction})
        comparisons = experiment.compare_batches(batches[0], batches, mask)
        self.assertEqual(comparisons["mean"]["critics"]["mc_factored"]["cross_batch_GAE_common_MC"]["count"], 2)
        self.assertEqual(comparisons["mean"]["raw_episode_noise"]["mc_common"]["episodes"], 4)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_disjoint_seed_roles_counts_and_dynamic_scheduler_dependency(self):
        full = spec.budget(preflight=False)
        self.assertEqual(full["native_episodes"] * 8, 1024)
        self.assertEqual(full["archive_episodes"] * 8, 256)
        self.assertEqual(spec.native_budget(preflight=False)["primitive_steps"] * 8, 1228800)
        self.assertEqual(full["actor_score_backward_batches"], 5 * full["actor_score_forward_batches"])
        for preflight in (True, False):
            seen = set()
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for b in roles["fresh_batches"] for s in b}
                self.assertFalse(seen & new)
                seen.update(new)
                old = spec.native.seed_roles(root, preflight=preflight)
                self.assertFalse(new & {s for seeds in old.values() for s in seeds})
                task = task_specification("unit_stage69", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 6144))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage69", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])


if __name__ == "__main__":
    unittest.main()
