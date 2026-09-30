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
from scripts import pointmaze_backtracking_stage60_spec as spec
from scripts.submit_pointmaze_backtracking_stage60_scheduleurm import task_specification
import test_pointmaze_state_baseline as baseline_data
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class BacktrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=3, lower_state_dim=3,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_learning_rate=.1))

    def std_step(self, model):
        model.lower_actor_optimizer.zero_grad()
        model.lower_actor.log_std.grad = torch.ones_like(model.lower_actor.log_std)
        model.lower_actor_optimizer.step()

    def test_all_accepted_preserves_exact_original_ppo(self):
        initial, batch, _ = baseline_data.StateBaselineTest().data()
        plain, bounded = copy.deepcopy(initial), copy.deepcopy(initial)
        expected = diagnostics.observed_update(plain, SimpleNamespace(lower=batch), level="lower", phase="train",
            root=310001, period=50, iteration=1, episode_count=3)
        row = experiment.guarded_update(bounded, SimpleNamespace(lower=batch), level="lower", root=310001,
            period=50, episode_count=3, budget=1e9, guard_type=experiment.BacktrackingKLGuard)
        g = row.pop("guard")
        self.assertEqual(row, expected)
        self.assertEqual(g["retained_actor_steps"], 6)
        self.assertEqual(g["candidate_evaluations"], 6)
        self.assertEqual(g["parameter_interpolation_trials"], 0)
        for name in ("lower_actor", "lower_value", "lower_actor_optimizer", "lower_value_optimizer"):
            torch.testing.assert_close(getattr(plain, name).state_dict(), getattr(bounded, name).state_dict(), atol=0, rtol=0)
        self.assertFalse(bounded.lower_actor_optimizer._optimizer_step_pre_hooks)
        self.assertFalse(bounded.lower_actor_optimizer._optimizer_step_post_hooks)

    def test_scaled_step_keeps_one_adam_moment_update_and_original_lr(self):
        model = self.model()
        self.std_step(model)
        full = copy.deepcopy(model)
        before = copy.deepcopy(model.lower_actor.state_dict())
        self.std_step(full)
        with experiment.BacktrackingKLGuard(model.lower_actor, model.lower_actor_optimizer, torch.zeros(4, 3), .002) as g:
            self.std_step(model)
        step = g.steps[0]
        self.assertTrue(step["accepted"])
        self.assertGreater(len(step["trials"]), 1)
        self.assertLess(step["accepted_scale"], 1.)
        self.assertLessEqual(step["deployed_kl"], .002)
        for name, parameter in model.lower_actor.named_parameters():
            torch.testing.assert_close(parameter, before[name] + step["accepted_scale"] *
                (full.lower_actor.state_dict()[name] - before[name]), atol=0, rtol=0)
        torch.testing.assert_close(model.lower_actor_optimizer.state_dict(), full.lower_actor_optimizer.state_dict(), atol=0, rtol=0)
        self.assertEqual(model.lower_actor_optimizer.param_groups[0]["lr"], .1)
        self.assertEqual(g.record()["rollback_state_checks"], 2 * (len(step["trials"]) - 1))

    def test_all_rejected_restores_nonempty_adam_and_reports_all_trials(self):
        model = self.model()
        self.std_step(model)
        actor, adam = copy.deepcopy(model.lower_actor.state_dict()), copy.deepcopy(model.lower_actor_optimizer.state_dict())
        with experiment.BacktrackingKLGuard(model.lower_actor, model.lower_actor_optimizer, torch.zeros(4, 3), 0.) as g:
            self.std_step(model)
        record = g.record()
        self.assertEqual(record["retained_actor_steps"], 0)
        self.assertEqual(record["candidate_evaluations"], 13)
        self.assertEqual(record["parameter_interpolation_trials"], 12)
        self.assertEqual(record["guard_distribution_passes"], 14)
        self.assertEqual(record["state_snapshot_calls"], 4)
        self.assertEqual(record["rollback_state_checks"], 26)
        torch.testing.assert_close(model.lower_actor.state_dict(), actor, atol=0, rtol=0)
        torch.testing.assert_close(model.lower_actor_optimizer.state_dict(), adam, atol=0, rtol=0)

    def test_reference_remains_fixed_across_accepted_prefix(self):
        model = self.model()
        with experiment.BacktrackingKLGuard(model.lower_actor, model.lower_actor_optimizer, torch.zeros(4, 3), .03) as g:
            self.std_step(model)
            reference = g.reference.mean.clone(), g.reference.stddev.clone()
            self.std_step(model)
            self.assertTrue(g.steps[0]["accepted"])
            self.assertTrue(g.steps[1]["accepted"])
            self.assertLess(g.steps[1]["accepted_scale"], 1.)
            self.assertLessEqual(g.steps[1]["deployed_kl"], .03)
            torch.testing.assert_close(g.reference.mean, reference[0], atol=0, rtol=0)
            torch.testing.assert_close(g.reference.stddev, reference[1], atol=0, rtol=0)
            self.assertAlmostEqual(g.candidate_kl()[0], g.steps[1]["deployed_kl"], places=15)

    def test_archive_pairing_and_variable_trial_accounting(self):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        helper = archive_data.UpdateDiagnosticsTest()
        model, predictor = helper.model(), helper.predictor
        source = {"config": json.loads(json.dumps(model.config.__dict__)), "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            file, diagnostic_file, rejection_file = [directory / stage / "result.json" for stage in ("stage57", "stage58", "stage59")]
            with patch.object(previous, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, source)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(spec.source, "source_result", return_value=file), \
                    patch.object(experiment.spec, "diagnostic_result", return_value=diagnostic_file), \
                    patch.object(spec, "diagnostic_result", return_value=diagnostic_file), \
                    patch.object(spec, "rejection_result", return_value=rejection_file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                previous.train(310001, preflight=True, output=file)
                diagnostics.replay(310001, preflight=True, output=diagnostic_file)
                experiment.replay(310001, preflight=True, output=rejection_file)
                with patch.object(joint, "_make_task", side_effect=AssertionError("archive comparison must not sample an environment")):
                    result = experiment.replay(310001, preflight=True, output=directory / "stage60/result.json", specification=spec)
                summary = experiment.aggregate([result], preflight=True, specification=spec)
        self.assertEqual(summary["cost"], spec.budget(preflight=True))
        self.assertEqual(summary["executed_optimizer_steps"], {"upper_actor": 6, "upper_value": 14, "lower_actor": 60, "lower_value": 100})
        for mutation in ("identity", "trials", "cost", "scale"):
            bad = copy.deepcopy(result)
            cell = bad["comparisons"]["50"]["joint_ppo"]
            g = cell["treatments"]["backtracking_kl"][0]["guard"]
            if mutation == "identity":
                cell["rejection_only_reproduction"] = "failed"
            elif mutation == "trials":
                g["candidate_evaluations"] += 1
            elif mutation == "cost":
                bad["cost"]["diagnostic_updates"] += 1
            else:
                g["steps"][0]["trials"][0]["scale"] = .5
            with self.assertRaises(ValueError):
                experiment.qualify(bad, preflight=True, specification=spec)

    def test_frozen_budget_and_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 4352)
        self.assertEqual(8 * b["diagnostic_updates"], 144)
        self.assertEqual(8 * b["diagnostic_distribution_passes"], 288)
        self.assertEqual((spec.KL_BUDGET, spec.BACKTRACK_FACTOR, spec.MAX_BACKTRACKS), (.02, .5, 12))
        task = task_specification("unit_stage60", 310011, preflight=False)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (9, 12288))


if __name__ == "__main__":
    unittest.main()
