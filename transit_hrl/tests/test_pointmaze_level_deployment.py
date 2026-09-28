import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_update_isolation as isolation
from freq_hrl.experiments import pointmaze_gate_deployment as gate
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_level_deployment_stage37_spec as spec
from scripts import analyze_pointmaze_level_deployment_stage37 as analyzer
from scripts.submit_pointmaze_level_deployment_stage37_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class LevelDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(37)
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False))
        return joint.make_model(source, "learned_history", root=310001)

    def rollout(self, model, *, sample=False, gate_sample=None, gate_seed=None, future_shift=0.):
        args = spec.source.arguments(310001, preflight=True)
        with patch.object(joint, "_make_task", return_value=DenseTask(future_shift)), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            return joint.rollout(model, args, "learned_history", seed=8394001, sample=sample,
                                 gate_sample=gate_sample, gate_seed=gate_seed, capture=True)

    def test_upper_lower_actor_and_critic_freeze_contract(self):
        for method in spec.METHODS[:4]:
            model = self.model()
            before = joint.inference_weights(model)
            batch, _, _ = self.rollout(model, sample=True)
            metrics = isolation.update_components(model, batch, spec.COMPONENTS[method])
            delta = isolation.change_norms(joint.inference_weights(model), before)
            for name, change in delta.items():
                enabled = name.rsplit("_", 1)[0] in spec.COMPONENTS[method]
                self.assertEqual(change > 0, enabled, (method, name))
                self.assertEqual(metrics.get(name + "_optimizer_steps", 0) > 0, enabled)

    def test_gate_sampling_is_reproducible_and_controllers_stay_deterministic(self):
        model = self.model()
        seed = spec.gate_seed(310001, 8394001, 0)
        with patch.object(model, "act_upper", wraps=model.act_upper) as upper, patch.object(
                model, "act_lower", wraps=model.act_lower) as lower, patch.object(
                model, "act_promotion", wraps=model.act_promotion) as promotion:
            _, row, raw = self.rollout(model, gate_sample=True, gate_seed=seed)
            self.assertTrue(all(call.kwargs["sample"] is False for call in upper.call_args_list))
            self.assertTrue(all(call.kwargs["sample"] is False for call in lower.call_args_list))
            self.assertTrue(all(call.kwargs["sample"] is True for call in promotion.call_args_list))
        torch.manual_seed(123)
        np.random.seed(123)
        _, replay, replay_raw = self.rollout(model, gate_sample=True, gate_seed=seed)
        self.assertEqual(row["gate_actions"], replay["gate_actions"])
        self.assertEqual(row["decision_steps"], replay["decision_steps"])
        np.testing.assert_array_equal(raw["reward"], replay_raw["reward"])
        _, _, shifted = self.rollout(model, gate_sample=True, gate_seed=seed, future_shift=2.)
        prefix = np.searchsorted(raw["gate_steps"], 100)
        np.testing.assert_array_equal(raw["gate_states"][:prefix], shifted["gate_states"][:prefix])
        np.testing.assert_array_equal(raw["gate_actions"][:prefix], shifted["gate_actions"][:prefix])

    def test_gate_probability_and_sampling_audit_detect_actual_action_changes(self):
        model = self.model()
        for sampled in (False, True):
            seed = spec.gate_seed(310001, 8394001, 0) if sampled else None
            _, row, raw = self.rollout(model, gate_sample=sampled, gate_seed=seed)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episode.npz"
                np.savez_compressed(path, **raw)
                gate.audit_probabilities(model, row, path)
                changed = copy.deepcopy(row)
                changed["gate_actions"][0] = 1 - changed["gate_actions"][0]
                with self.assertRaises(AssertionError):
                    gate.audit_probabilities(model, changed, path)

    def test_main_driver_reuses_trainer_with_stage37_roles_and_audit(self):
        model = self.model()
        args = spec.source.arguments(310001, preflight=True)
        cell = {"selected_checkpoint_iteration": 0, "factual_row": {"decision_steps": list(range(0, args.horizon, 50))}}
        with tempfile.TemporaryDirectory() as directory, patch.object(isolation, "load_controller", return_value=(model, cell, {})), \
                patch.object(isolation, "ProcessPoolExecutor", ImmediatePool), \
                patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            result = isolation.train(310001, "lower_only", preflight=True,
                                     output=Path(directory) / "cell" / "result.json", specification=spec)
            self.assertEqual(result["protocol"], spec.EXPERIMENT_PROTOCOL)
            self.assertEqual(result["optimizer_steps"]["lower_actor_optimizer_steps"] > 0, True)
            self.assertEqual(result["trained_parameter_change_norms"]["upper_actor"], 0)
            self.assertEqual(result["trained_parameter_change_norms"]["promotion_actor"], 0)
            isolation.audit_result(result, raw_path=Path(directory) / "cell_raw", specification=spec)

    def test_gate_driver_evaluates_all_sources_modes_and_streams_without_updates(self):
        model = self.model()
        def cached(root, method, preflight):
            return model, str(Path("/cached") / method / "final.pt"), 2
        with tempfile.TemporaryDirectory() as directory, patch.object(gate, "load_cached", side_effect=cached), \
                patch.object(gate, "ProcessPoolExecutor", ImmediatePool), \
                patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            before = joint.inference_weights(model)
            result = gate.evaluate(310001, preflight=True, output=Path(directory) / "cell" / "result.json")
            audit = gate.audit_result(result, raw_path=Path(directory) / "cell_raw")
            self.assertEqual(audit["episodes"], 18)
            self.assertEqual(len(audit["replays"]), 6)
            self.assertEqual(result["budget"]["training_primitive_steps"], 0)
            self.assertTrue(all(delta == 0 for delta in isolation.change_norms(joint.inference_weights(model), before).values()))
            changed = copy.deepcopy(result)
            changed["evaluation_rows"]["gate_only"]["sampled"].pop()
            with self.assertRaisesRegex(ValueError, "roster incomplete"):
                gate.audit_result(changed, raw_path=Path(directory) / "cell_raw")

    def test_rosters_costs_and_scheduler_sources(self):
        seen = set()
        for preflight in (False, True):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = [s for values in roles.values() for s in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                old = spec.previous.seed_roles(root, preflight=preflight)
                self.assertFalse(set(seeds).intersection(s for values in old.values() for s in values))
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"] +
                         8 * spec.gate_budget(preflight=False)["total_primitive_steps"], 58800000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 9)
        for method in (*spec.METHODS, spec.GATE_TASK):
            task = task_specification("unit_stage37", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])
            self.assertIn("&& printf '%s\\n' 'Training complete: result.json written'", task["cmd"])

    def test_related_submitters_use_scheduler_recognized_completion_marker(self):
        files = list((spec.ROOT / "scripts").glob("submit_*scheduleurm.py"))
        matched = 0
        for path in files:
            for line in path.read_text().splitlines():
                if "complete: result.json written" in line:
                    self.assertIn("Training complete: result.json written", line, str(path))
                    matched += 1
        self.assertGreater(matched, 0)

    def test_sampling_gain_is_not_reclassified_as_training_gain(self):
        main, cached = [], []
        for root in spec.OPTIMIZER_ROOTS:
            def row(value):
                return {"episode_return": value, "tracking_squared_error_integral": 2.,
                        "upper_inference_calls": 24., "charged_utility": value - 24.}
            for method, value in zip(spec.METHODS, (100., 80., 105., 85., 110.)):
                main.append({"root": root, "method": method, "evaluation_rows": {
                    "final": [row(value)], "selected": [row(200.)]}})
            cached.append({"root": root, "evaluation_rows": {
                method: {mode: [row(value)] for mode, value in zip(spec.GATE_MODES, values)}
                for method, values in zip(spec.GATE_SOURCES, ((90., 100.), (80., 90.), (50., 70.)))}})
        level = isolation.aggregate(main, specification=spec)
        self.assertEqual(level["primary_endpoints"]["upper_only:frozen:return"]["effect"], "negative")
        self.assertEqual(level["primary_endpoints"]["lower_only:frozen:return"]["effect"], "positive")
        self.assertEqual(level["primary_endpoints"]["interaction:return"]["mean"], 0)
        deployment = gate.aggregate(cached)
        self.assertEqual(deployment["primary_endpoints"]["gate_only:sampled_threshold:return"]["effect"], "positive")
        self.assertEqual(deployment["primary_endpoints"]["gate_only:frozen:sampled:return"]["effect"], "negative")
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            gate.aggregate(cached[:-1])

    def test_incomplete_gate_audit_never_writes_complete_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(spec, "ROOT", Path(directory)), \
                patch.object(analyzer, "analyze_training", return_value={"status": "complete"}):
            with self.assertRaises(FileNotFoundError):
                analyzer.analyze("incomplete", preflight=True)
            self.assertFalse((Path(directory) / "results/incomplete/qualification_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
