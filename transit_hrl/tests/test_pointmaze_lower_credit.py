import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_update_isolation as isolation
from freq_hrl.experiments.pointmaze_lower_credit import audit_credit_batch, audit_training_credit
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_lower_credit_stage38_spec as spec
from scripts import analyze_pointmaze_lower_credit_stage38 as analyzer
from scripts.submit_pointmaze_lower_credit_stage38_scheduleurm import task_specification
from test_pointmaze_joint_renewal import CountedController, DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class LowerCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(38)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False))

    def rollout(self, mode, *, future_shift=0.):
        args = spec.source.arguments(310001, preflight=True)
        with patch.object(joint, "_make_task", return_value=DenseTask(future_shift)), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            return joint.rollout(CountedController(), args, "learned_history", seed=1,
                                 sample=True, capture=True, lower_credit=mode)

    def test_reward_and_boundary_factorial_changes_only_lower_credit(self):
        args = spec.source.arguments(310001, preflight=True)
        batches = {}
        baseline_raw = None
        for mode in spec.METHODS[1:]:
            batch, row, raw = self.rollout(mode)
            audit_credit_batch(batch, row, raw, args=args, mode=mode)
            batches[mode] = batch
            if baseline_raw is None:
                baseline_raw = raw
            for key in ("physical", "action", "reward", "decision_steps", "gate_actions"):
                np.testing.assert_array_equal(raw[key], baseline_raw[key])
            count = len(row["decision_steps"]) if mode.endswith("_option") else 1
            self.assertEqual(int(batch.lower.done.sum()), count)
            if mode.startswith("task_"):
                np.testing.assert_array_equal(batch.lower.reward, raw["reward"].astype(np.float32))
        baseline = batches["intrinsic_option"]
        for batch in batches.values():
            for level in ("upper", "promotion"):
                for key in ("state", "action", "reward", "duration", "done", "old_logp", "old_value"):
                    np.testing.assert_array_equal(getattr(getattr(batch, level), key), getattr(getattr(baseline, level), key))
        np.testing.assert_array_equal(batches["intrinsic_episode"].lower.reward, baseline.lower.reward)
        self.assertFalse(np.array_equal(baseline.lower.reward, batches["task_option"].lower.reward))

    def test_default_is_the_unchanged_intrinsic_option_recipe(self):
        args = spec.source.arguments(310001, preflight=True)
        with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            default = joint.rollout(CountedController(), args, "learned_history", seed=1, sample=True)[0]
            explicit = joint.rollout(CountedController(), args, "learned_history", seed=1,
                                     sample=True, lower_credit="intrinsic_option")[0]
            for key in ("reward", "done", "old_value", "action"):
                np.testing.assert_array_equal(getattr(default.lower, key), getattr(explicit.lower, key))

    def test_episode_credit_carries_gae_across_replanning_not_across_episodes(self):
        model = self.controller()
        option, _, _ = self.rollout("task_option")
        episode, _, _ = self.rollout("task_episode")
        signal = np.zeros_like(option.lower.reward)
        signal[100] = 1.
        values = np.zeros_like(signal)
        args = (signal, option.lower.done, option.lower.duration, values)
        option_advantage, _ = model._gae(*args)
        episode_advantage, _ = model._gae(signal, episode.lower.done, episode.lower.duration, values)
        self.assertEqual(option_advantage[99], 0.)
        self.assertGreater(episode_advantage[99], 0.)
        combined = isolation.concat_hierarchical_batches([episode, episode])
        self.assertEqual(combined.lower.done[len(signal) - 1], 1.)

    def test_native_credit_audit_detects_wrong_reward_and_option_cuts(self):
        args = spec.source.arguments(310001, preflight=True)
        batch, row, raw = self.rollout("task_episode")
        changed = copy.deepcopy(batch)
        changed.lower.reward[0] += 1.
        with self.assertRaises(AssertionError):
            audit_credit_batch(changed, row, raw, args=args, mode="task_episode")
        changed = copy.deepcopy(batch)
        changed.lower.done[99] = 1.
        with self.assertRaises(AssertionError):
            audit_credit_batch(changed, row, raw, args=args, mode="task_episode")

    def test_reward_changes_do_not_change_causal_action_prefix(self):
        for mode in spec.METHODS[1:]:
            _, _, original = self.rollout(mode)
            _, _, future = self.rollout(mode, future_shift=2.)
            for key in ("action", "physical", "subgoal"):
                np.testing.assert_array_equal(original[key][:100], future[key][:100])

    def test_driver_records_actual_training_credit_and_freezes_upper_gate(self):
        args = spec.source.arguments(310001, preflight=True)
        cell = {"selected_checkpoint_iteration": 0, "factual_row": {"decision_steps": list(range(0, args.horizon, 50))}}
        for method in spec.METHODS:
            with tempfile.TemporaryDirectory() as directory, patch.object(isolation, "load_controller", return_value=(self.controller(), cell, {})), \
                    patch.object(isolation, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                result = isolation.train(310001, method, preflight=True, output=Path(directory) / "cell/result.json",
                                         specification=spec, lower_credit=spec.LOWER_CREDIT[method])
                isolation.audit_result(result, raw_path=Path(directory) / "cell_raw", specification=spec)
                audit_training_credit(result, specification=spec)
                for level in ("upper", "promotion"):
                    for suffix in ("actor", "value"):
                        self.assertEqual(result["trained_parameter_change_norms"][level + "_" + suffix], 0.)
                self.assertEqual(result["initial_credit_probe"]["mode"], spec.LOWER_CREDIT[method])
                changed = copy.deepcopy(result)
                changed["training_credit"][0]["done_count"] += 1
                with self.assertRaisesRegex(ValueError, "boundary accounting"):
                    audit_training_credit(changed, specification=spec)

    def test_rosters_budgets_and_scheduler_marker(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = [s for values in spec.seed_roles(root, preflight=preflight).values() for s in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                old = {s for values in spec.previous.seed_roles(root, preflight=preflight).values() for s in values}
                self.assertFalse(set(seeds).intersection(old))
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"], 54192000)
        self.assertEqual(spec.verification_budget(preflight=False)["total_primitive_steps"], 144000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 7)
        for method in spec.METHODS:
            task = task_specification("unit_stage38", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("Training complete: result.json written", task["cmd"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])

    def test_factorial_effects_do_not_use_selected_cohort(self):
        results = []
        for root in spec.OPTIMIZER_ROOTS:
            for method, value in zip(spec.METHODS, (100., 100., 110., 120., 140.)):
                row = {"episode_return": value, "tracking_squared_error_integral": 2.,
                       "upper_inference_calls": 24., "charged_utility": value - 24.}
                results.append({"root": root, "method": method, "evaluation_rows": {
                    "final": [row], "selected": [{**row, "episode_return": 999.}]}})
        summary = isolation.aggregate(results, specification=spec)
        expected = (0., 10., 20., 40., 25., 15., 10.)
        for key, value in zip(spec.ENDPOINTS, expected):
            self.assertEqual(summary["primary_endpoints"][key]["mean"], value)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            isolation.aggregate(results[:-1], specification=spec)

    def test_incomplete_credit_audit_does_not_write_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(spec, "ROOT", Path(directory)), \
                patch.object(analyzer, "collect_results", return_value={"status": "complete"}):
            with self.assertRaises(FileNotFoundError):
                analyzer.analyze("incomplete", preflight=True)
            self.assertFalse((Path(directory) / "results/incomplete/qualification_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
