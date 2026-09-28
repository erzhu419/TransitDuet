import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_critic_calibration_stage39_spec as spec
from scripts import analyze_pointmaze_critic_calibration_stage39 as analyzer
from scripts.submit_pointmaze_critic_calibration_stage39_scheduleurm import task_specification
from test_pointmaze_joint_renewal import CountedController, DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class CalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(39)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, epochs=1, minibatch_size=128,
            upper_learning_rate=3e-4, lower_learning_rate=3e-4))

    def test_mc_targets_respect_option_terminals_and_durations(self):
        batch = SimpleNamespace(size=4, reward=np.array([1., 2., 3., 4.]),
                                done=np.array([0., 1., 0., 1.]), duration=np.array([1, 1, 2, 1]))
        np.testing.assert_array_equal(calibration.monte_carlo_returns(batch, .5), [2., 2., 4., 4.])

    def test_critic_only_uses_existing_ppo_without_touching_actor_optimizer(self):
        model = joint.make_model(self.controller(), "learned_history", root=310001)
        args = spec.source.arguments(310001, preflight=True)
        with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            batch = joint.rollout(model, args, "learned_history", seed=1, sample=True)[0].lower
        before, reference = joint.inference_weights(model), copy.deepcopy(model.lower_actor)
        np.random.seed(39)
        metrics = calibration.lower_update(model, batch, "critic")
        delta = calibration.change_norms(joint.inference_weights(model), before)
        self.assertGreater(delta["lower_value"], 0.)
        for name in delta:
            if name != "lower_value":
                self.assertEqual(delta[name], 0.)
        self.assertFalse(model.lower_actor_optimizer.state)
        self.assertEqual(metrics["lower_actor_optimizer_steps"], 0.)
        self.assertEqual(metrics["lower_value_optimizer_steps"], 3.)
        self.assertEqual(calibration.policy_drift(model.lower_actor, reference, batch.state)["gaussian_kl"], 0.)
        np.random.seed(39)
        calibration.lower_update(model, batch, "actor_critic")
        self.assertGreater(calibration.policy_drift(model.lower_actor, reference, batch.state)["gaussian_kl"], 0.)

    def test_lower_sampling_does_not_sample_upper_or_gate(self):
        class RecordedController(CountedController):
            def __init__(self):
                super().__init__()
                self.samples = {level: [] for level in ("upper", "lower", "gate")}

            def act_upper(self, state, sample):
                self.samples["upper"].append(sample)
                return super().act_upper(state, sample)

            def act_lower(self, state, sample):
                self.samples["lower"].append(sample)
                return super().act_lower(state, sample)

            def act_promotion(self, state, sample):
                self.samples["gate"].append(sample)
                return super().act_promotion(state, sample)

        args, model = spec.source.arguments(310001, preflight=True), RecordedController()
        with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            batch, row, _ = joint.rollout(model, args, "learned_history", seed=1, sample=False, lower_sample=True)
        self.assertIsNone(batch)
        self.assertTrue(row["lower_sample"])
        self.assertFalse(row["gate_sample"])
        self.assertTrue(all(model.samples["lower"]))
        self.assertFalse(any(model.samples["upper"]))
        self.assertFalse(any(model.samples["gate"]))

    def test_all_cells_native_replay_freezes_and_accounting(self):
        controller = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path, source_checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, source_checkpoint)
            source_cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(source_checkpoint),
                           "factual_row": {"decision_steps": [0, 100, 200]}}
            source_path.write_text(json.dumps({"cells": [source_cell]}))
            with patch.object(spec, "ROOT", directory), patch.object(spec, "source_result", return_value=source_path), \
                    patch.object(calibration, "load_controller", return_value=(controller, source_cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                cells = {}
                for method in spec.METHODS:
                    output = directory / "results/test/cells" / method / "replicate_310001/result.json"
                    result = calibration.train(310001, method, preflight=True, output=output)
                    cells[method] = result
                    raw = output.parent.with_name(output.parent.name + "_raw")
                    calibration.audit_result(result, raw_path=raw)
                    expected_actor = 0 if method == "frozen" else 6
                    expected_value = 12 if method.endswith("_calibrated") else expected_actor
                    self.assertEqual(result["optimizer_steps"], {"actor_optimizer_steps": expected_actor,
                                                                "value_optimizer_steps": expected_value})
                    self.assertEqual(result["snapshots"]["2"]["parameter_change_norms"]["lower_actor"], 0.)
                    for mode in spec.MODES:
                        for row0, row2 in zip(result["snapshots"]["0"]["evaluation_rows"][mode],
                                               result["snapshots"]["2"]["evaluation_rows"][mode]):
                            self.assertEqual(row0["episode_return"], row2["episode_return"])
                    changed = copy.deepcopy(result)
                    changed["training"][0]["actor_optimizer_steps"] = 1
                    with self.assertRaisesRegex(ValueError, "update contract"):
                        calibration.audit_result(changed, raw_path=raw)
                    changed = copy.deepcopy(result)
                    changed["snapshots"]["0"]["evaluation_rows"]["lower_sampled"][0]["gate_sample"] = True
                    with self.assertRaisesRegex(ValueError, "deployment sampling"):
                        calibration.audit_result(changed, raw_path=raw)
                summary = analyzer.analyze("test", preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(len(summary["checkpoint_replays"]), 40)
                self.assertEqual(len(summary["probe_replays"]), 10)
                self.assertEqual(sum(a["episodes"] for a in summary["audits"]), 80)
                self.assertEqual(summary["method_cost"]["primitive_steps"], 33000)
                self.assertEqual(summary["verification_cost"]["primitive_steps"], 15000)
                self.assertNotIn("primary_endpoints", summary["aggregate"])
                for source in ("intrinsic", "task"):
                    for row_a, row_b in zip(cells[source + "_delayed"]["training"][:2],
                                            cells[source + "_calibrated"]["training"][:2]):
                        self.assertEqual(row_a["reward_sum"], row_b["reward_sum"])
                (directory / "results/test/qualification_summary.json").unlink()
                (directory / "results/test/cells/task_calibrated/replicate_310001/result.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    analyzer.analyze("test", preflight=True)
                self.assertFalse((directory / "results/test/qualification_summary.json").exists())

    def test_final_and_first_update_contrasts_ignore_sampled_curves(self):
        results = []
        final = dict(zip(spec.METHODS, (100., 105., 110., 90., 95.)))
        first = dict(zip(spec.METHODS, (100., 110., 112., 110., 105.)))
        for root in spec.roots(preflight=False):
            for method in spec.METHODS:
                stages = {}
                for iteration in spec.snapshots(preflight=False):
                    value = (final if iteration == 32 else first)[method]
                    row = {"episode_return": value, "tracking_squared_error_integral": 1.,
                           "upper_inference_calls": 24., "charged_utility": value - 24.}
                    stages[str(iteration)] = {"evaluation_rows": {"deterministic": [row],
                                               "lower_sampled": [{**row, "episode_return": 999.}]}}
                results.append({"root": root, "method": method, "snapshots": stages})
        summary = calibration.aggregate(results, preflight=False)
        for key, value in zip(spec.ENDPOINTS, (5., 10., -10., -5., 5., 5., 2., -5.)):
            self.assertEqual(summary["primary_endpoints"][key]["mean"], value)
            self.assertEqual(summary["primary_endpoints"][key]["ci"], [value, value])
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            calibration.aggregate(results[:-1], preflight=False)

    def test_frozen_rosters_costs_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = [s for values in spec.seed_roles(root, preflight=preflight).values() for s in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                old = {s for values in spec.previous.seed_roles(root, preflight=preflight).values() for s in values}
                self.assertFalse(set(seeds).intersection(old))
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"], 20064000)
        self.assertEqual(spec.verification_budget(preflight=False)["total_primitive_steps"], 576000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        for method in spec.METHODS:
            task = task_specification("unit_stage39", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("Training complete: result.json written", task["cmd"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])


if __name__ == "__main__":
    unittest.main()
