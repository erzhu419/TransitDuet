import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_first_update as experiment
from freq_hrl.experiments import pointmaze_update_diagnostics as diagnostics
from freq_hrl.experiments import pointmaze_matched_upper as previous
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_first_update_stage59_spec as spec
from scripts.submit_pointmaze_first_update_stage59_scheduleurm import task_specification
import test_pointmaze_state_baseline as baseline_data
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class FirstUpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def update(self, model, batch, *, budget):
        return experiment.guarded_update(model, SimpleNamespace(lower=batch), level="lower",
            root=310001, period=50, episode_count=3, budget=budget)

    def test_all_accepted_hook_preserves_original_ppo_exactly(self):
        initial, batch, _ = baseline_data.StateBaselineTest().data()
        plain, bounded = copy.deepcopy(initial), copy.deepcopy(initial)
        expected = diagnostics.observed_update(plain, SimpleNamespace(lower=batch), level="lower", phase="train",
            root=310001, period=50, iteration=1, episode_count=3)
        observed = self.update(bounded, batch, budget=1e9)
        guard = observed.pop("guard")
        self.assertEqual(observed, expected)
        self.assertEqual(guard["retained_actor_steps"], 6)
        self.assertEqual(guard["rejected_actor_steps"], 0)
        for name in ("lower_actor", "lower_value", "lower_actor_optimizer", "lower_value_optimizer"):
            torch.testing.assert_close(getattr(plain, name).state_dict(), getattr(bounded, name).state_dict(), atol=0, rtol=0)
        self.assertFalse(bounded.lower_actor_optimizer._optimizer_step_pre_hooks)
        self.assertFalse(bounded.lower_actor_optimizer._optimizer_step_post_hooks)

    def test_all_rejected_restore_nonempty_adam_and_keep_original_critic(self):
        model, batch, _ = baseline_data.StateBaselineTest().data()
        self.update(model, batch, budget=1e9)
        with torch.no_grad():
            batch.old_logp = model.lower_actor.log_prob_entropy(torch.from_numpy(batch.state), torch.from_numpy(batch.action))[0].numpy()
            batch.old_value = model.lower_value(torch.from_numpy(batch.value_state)).numpy()
        before = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in ("lower_actor", "lower_actor_optimizer")}
        self.assertTrue(model.lower_actor_optimizer.state)
        plain = copy.deepcopy(model)
        diagnostics.observed_update(plain, SimpleNamespace(lower=batch), level="lower", phase="train",
            root=310001, period=50, iteration=1, episode_count=3)
        row = self.update(model, batch, budget=0.)
        self.assertEqual(row["guard"]["retained_actor_steps"], 0)
        self.assertEqual(row["guard"]["rejected_actor_steps"], 6)
        self.assertEqual(row["guard"]["guard_distribution_passes"], 7)
        self.assertEqual(row["guard"]["state_snapshot_calls"], 12)
        self.assertEqual(row["guard"]["rollback_state_checks"], 12)
        self.assertEqual(row["kl_mean"], 0.)
        self.assertEqual(row["mean_action_change_rms"], 0.)
        for name, state in before.items():
            torch.testing.assert_close(getattr(model, name).state_dict(), state, atol=0, rtol=0)
        for name in ("lower_value", "lower_value_optimizer"):
            torch.testing.assert_close(getattr(model, name).state_dict(), getattr(plain, name).state_dict(), atol=0, rtol=0)

    def test_reference_stays_at_sampling_policy_and_rejected_prefix_restores_adam(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=3, lower_state_dim=3,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_learning_rate=.1))
        actor, optimizer = model.lower_actor, model.lower_actor_optimizer
        with experiment.ConditionalKLGuard(actor, optimizer, torch.zeros(4, 3), .03) as guard:
            optimizer.zero_grad()
            actor.log_std.grad = torch.ones_like(actor.log_std)
            optimizer.step()
            self.assertTrue(guard.steps[0]["accepted"])
            first = copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad()
            actor.log_std.grad = torch.ones_like(actor.log_std)
            optimizer.step()
            self.assertFalse(guard.steps[1]["accepted"])
            self.assertEqual(guard.steps[1]["deployed_kl"], guard.steps[0]["candidate_kl"])
            torch.testing.assert_close(actor.state_dict(), first[0], atol=0, rtol=0)
            torch.testing.assert_close(optimizer.state_dict(), first[1], atol=0, rtol=0)

    def test_hook_removed_when_update_raises(self):
        model, batch, _ = baseline_data.StateBaselineTest().data()
        with self.assertRaisesRegex(RuntimeError, "actual update failed"):
            with experiment.ConditionalKLGuard(model.lower_actor, model.lower_actor_optimizer, torch.from_numpy(batch.state), .02):
                raise RuntimeError("actual update failed")
        self.assertFalse(model.lower_actor_optimizer._optimizer_step_pre_hooks)
        self.assertFalse(model.lower_actor_optimizer._optimizer_step_post_hooks)

    def test_archived_pipeline_pairing_accounting_and_qualification_failures(self):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        helper = archive_data.UpdateDiagnosticsTest()
        model, predictor = helper.model(), helper.predictor
        source = {"config": json.loads(json.dumps(model.config.__dict__)), "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            file, diagnostic_file = directory / "stage57/result.json", directory / "stage58/result.json"
            with patch.object(previous, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, source)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(spec.source, "source_result", return_value=file), \
                    patch.object(spec, "diagnostic_result", return_value=diagnostic_file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                previous.train(310001, preflight=True, output=file)
                diagnostics.replay(310001, preflight=True, output=diagnostic_file)
                with patch.object(joint, "_make_task", side_effect=AssertionError("archive comparison must not sample an environment")):
                    result = experiment.replay(310001, preflight=True, output=directory / "stage59/result.json")
                summary = experiment.aggregate([result], preflight=True)
        self.assertEqual(summary["status"], "preflight_passed")
        self.assertEqual(summary["cost"], spec.budget(preflight=True))
        self.assertEqual(summary["executed_optimizer_steps"], {"upper_actor": 4, "upper_value": 12, "lower_actor": 40, "lower_value": 80})
        for mutation in ("cost", "steps", "guard", "identity", "roster"):
            bad = copy.deepcopy(result)
            d = bad["comparisons"]["50"]["joint_ppo"]["treatments"]["conditional_kl"][0]
            if mutation == "cost":
                bad["cost"]["archive_episodes"] += 1
            elif mutation == "steps":
                d["optimizer_steps"]["upper_actor_optimizer_steps"] += 1
            elif mutation == "guard":
                d["guard"]["guard_distribution_passes"] += 1
            elif mutation == "identity":
                bad["comparisons"]["50"]["joint_ppo"]["critic_networks_and_Adam_pair"] = "failed"
            else:
                del bad["comparisons"]["100"]
            with self.assertRaises(ValueError):
                experiment.qualify(bad, preflight=True)
        with self.assertRaises(ValueError):
            experiment.aggregate([], preflight=True)

    def test_frozen_budget_and_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 4352)
        self.assertEqual(8 * b["reconstructed_lower_calls"], 5222400)
        self.assertEqual(8 * b["reconstructed_upper_calls"], 78336)
        self.assertEqual(8 * b["warmup_critic_updates"], 1024)
        self.assertEqual(8 * b["diagnostic_updates"], 96)
        self.assertEqual(spec.KL_BUDGET, .02)
        for preflight in (True, False):
            self.assertEqual(spec.options(preflight=preflight)["learning_iterations"], 1)
            task = task_specification("unit_stage59", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertEqual(task["ram_mb"], 4096 if preflight else 12288)


if __name__ == "__main__":
    unittest.main()
