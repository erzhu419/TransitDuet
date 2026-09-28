from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments.pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_joint_renewal_stage35_spec as spec
from scripts import pointmaze_response_deployment_stage34_spec as previous
from scripts.submit_pointmaze_joint_renewal_stage35_scheduleurm import task_specification
from test_pointmaze_temporal_plan import FakeTask


class DenseTask(FakeTask):
    def step(self, action):
        before_target = self.observation().target.copy()
        obs, _, _, _, _ = super().step(action)
        distance = float(np.linalg.norm(obs.achieved_goal - before_target))
        return obs, float(np.exp(-distance)), False, False, {"tracking_distance": distance}


class CountedController:
    def __init__(self, gate_action=0.):
        self.config = SimpleNamespace(gamma=.995)
        self.upper_calls = self.lower_calls = self.gate_calls = 0
        self.gate_action = gate_action

    def reset_recurrent_inference(self):
        pass

    def act_upper(self, state, sample):
        self.upper_calls += 1
        return {"action": np.array([.2, -.1]), "logp": 0., "value": 0.}

    def act_lower(self, state, sample):
        self.lower_calls += 1
        return {"action": np.tanh(state[4:6]), "logp": 0., "value": 0.}

    def act_promotion(self, state, sample):
        self.gate_calls += 1
        return {"action": self.gate_action, "logp": 0., "value": 0.}


class JointRenewalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fake_rollout(self, model, method, *, sample=False, future_shift=0., horizon=200):
        args = spec.source.arguments(310001, preflight=True)
        args.horizon = horizon
        with patch.object(joint, "_make_task", return_value=DenseTask(future_shift)), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            return joint.rollout(model, args, method, seed=1, sample=sample, capture=True)

    def test_plan_called_only_on_execution_and_lower_every_step(self):
        for gate_action, expected in ((0., [0, 100]), (1., list(range(0, 200, 25)))):
            model = CountedController(gate_action)
            _, row, raw = self.fake_rollout(model, "learned_history")
            self.assertEqual(row["decision_steps"], expected)
            self.assertEqual(row["upper_inference_calls"], model.upper_calls)
            self.assertEqual(model.lower_calls, 200)
            self.assertEqual(row["candidate_preview_calls"], 0)
            self.assertEqual(row["gate_inference_calls"], model.gate_calls)
            self.assertAlmostEqual(row["charged_utility"], raw["reward"].sum() - len(expected))

    def test_native_credit_and_lower_option_boundaries(self):
        model = CountedController()
        batch, row, raw = self.fake_rollout(model, "learned_history", sample=True)
        np.testing.assert_array_equal(batch.upper.duration, [100, 100])
        np.testing.assert_array_equal(np.flatnonzero(batch.lower.done), [99, 199])
        self.assertEqual(batch.lower.size, 200)
        expected = raw["reward"].copy()
        expected[row["decision_steps"]] -= spec.CALL_COST
        for index, start in enumerate(row["decision_steps"]):
            self.assertAlmostEqual(batch.upper.reward[index], np.dot(expected[start:start + 100], .995 ** np.arange(100)), places=5)
        edges = [*row["gate_steps"], 200]
        for index, (start, end) in enumerate(zip(edges, edges[1:])):
            self.assertEqual(batch.promotion.duration[index], end - start)
            self.assertAlmostEqual(batch.promotion.reward[index], np.dot(expected[start:end], .995 ** np.arange(end - start)), places=5)
        self.assertEqual(batch.promotion.done[-1], 1.)

    def test_fixed_baselines_do_not_pay_gate_or_preview(self):
        for method, period in (("fixed50", 50), ("fixed100", 100)):
            model = CountedController()
            batch, row, _ = self.fake_rollout(model, method, sample=True)
            self.assertEqual(row["decision_steps"], list(range(0, 200, period)))
            self.assertEqual(model.gate_calls, 0)
            self.assertIsNone(batch.promotion)
            np.testing.assert_array_equal(np.flatnonzero(batch.lower.done), np.arange(period - 1, 200, period))

    def test_future_changes_do_not_change_observed_gate_prefix(self):
        _, _, factual = self.fake_rollout(CountedController(), "learned_history")
        _, _, changed = self.fake_rollout(CountedController(), "learned_history", future_shift=2.)
        np.testing.assert_array_equal(factual["gate_states"][:3], changed["gate_states"][:3])
        self.assertFalse(np.array_equal(factual["gate_states"][-1], changed["gate_states"][-1]))

    def test_current_control_same_width_and_latest_measurement(self):
        args = spec.source.arguments(310001, preflight=True)
        history = PointMazeRegimeFeatureBuilder(time_scale=joint.scale_for(args))
        task = DenseTask()
        observation = task.reset()
        history.reset(observation)
        for _ in range(25):
            observation = task.step(np.zeros(2))[0]
            history.update(observation)
        common = dict(age=25, step=25, horizon=200)
        h = joint.gate_state(observation, history, np.zeros(2), current_only=False, **common)
        c = joint.gate_state(observation, history, np.zeros(2), current_only=True, **common)
        self.assertEqual(h.shape, c.shape)
        np.testing.assert_array_equal(c[6:-4].reshape(64, 6), np.tile(observation.task_measurement.astype(np.float32), (64, 1)))
        np.testing.assert_array_equal(h[-10:], c[-10:])
        self.assertFalse(np.array_equal(h, c))

    def test_reuses_smdp_trainer_and_updates_all_three_actors(self):
        args = spec.source.arguments(310001, preflight=True)
        dimension = 6 + joint.scale_for(args).history_steps * 6
        torch.manual_seed(35)
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=dimension, lower_state_dim=dimension, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, upper_learning_rate=3e-4, lower_learning_rate=3e-4))
        model = joint.make_model(source, "learned_history", root=310001)
        current = joint.make_model(source, "learned_current", root=310001)
        weights = joint.inference_weights(model)
        for name in weights:
            for key in weights[name]:
                np.testing.assert_array_equal(weights[name][key].numpy(), joint.inference_weights(current)[name][key].numpy())
        for name in ("upper_actor", "lower_actor", "upper_value", "lower_value"):
            for key in getattr(source, name).state_dict():
                np.testing.assert_array_equal(weights[name][key], getattr(source, name).state_dict()[key])
        self.assertFalse(model.upper_actor_optimizer.state)
        batch, _, _ = self.fake_rollout(model, "learned_history", sample=True)
        update = model.update(batch)
        updated = joint.inference_weights(model)
        for level in ("upper", "lower", "promotion"):
            self.assertGreater(update[level + "_actor_optimizer_steps"], 0)
            self.assertTrue(any(not torch.equal(value, updated[level + "_actor"][key]) for key, value in weights[level + "_actor"].items()))

    def test_rosters_disjoint_and_full_equal_budget(self):
        all_seeds = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = [seed for values in roles.values() for seed in values]
                self.assertEqual(len(set(seeds)), len(seeds))
                self.assertFalse(all_seeds.intersection(seeds))
                all_seeds.update(seeds)
                self.assertFalse(set(seeds).intersection(previous.evaluation_paths(root, preflight=preflight)))
                self.assertFalse(set(seeds).intersection(seed for values in spec.source.seed_roles(root, preflight=preflight).values() for seed in values))
        self.assertEqual(spec.budget(preflight=False)["total_primitive_steps"] * 32, 42124800)

    def test_source_only_scheduler_staging_and_dynamic_cpu_pool(self):
        for method in spec.METHODS:
            task = task_specification("unit_joint", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])
            self.assertFalse(any("/results/" in path for path in task["stage_input_paths"]))

    def test_auditor_reads_each_compressed_array_only_once(self):
        root, method = 310001, "fixed100"
        roles, budget = spec.seed_roles(root, preflight=True), spec.budget(preflight=True)
        horizon = spec.source.arguments(root, preflight=True).horizon
        _, row, raw = self.fake_rollout(CountedController(), method, horizon=horizon)
        rows = [{**row, "seed": seed} for seed in roles["evaluation"]]
        result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                  "root": root, "method": method, "preflight": True, "options": spec.options(preflight=True),
                  "seed_roles": roles, "budget": budget, "evaluation_rows": rows,
                  "optimizer_steps": {"upper_actor_optimizer_steps": 1, "lower_actor_optimizer_steps": 1},
                  "trained_parameter_change_norms": {"upper_actor": 1., "lower_actor": 1.},
                  "selection_history": [{"iteration": i, "utility": i, "ise": 1.} for i in (0, 1, 2)], "selected_iteration": 2,
                  "inference_counts": {phase: {"lower_inference_calls": budget[field]}
                                       for phase, field in (("train", "training_primitive_steps"), ("selection", "selection_primitive_steps"),
                                                            ("eval", "evaluation_primitive_steps"), ("factual_replay", "factual_replay_primitive_steps"))}}
        with tempfile.TemporaryDirectory() as directory:
            for seed in roles["evaluation"]:
                np.savez_compressed(Path(directory) / f"episode_{seed}.npz", **raw)
            with np.load(Path(directory) / f"episode_{roles['evaluation'][0]}.npz") as archive:
                archive_type = type(archive)
            original = archive_type.__getitem__
            reads = Counter()

            def counted(archive, key):
                reads[key] += 1
                return original(archive, key)

            with patch.object(archive_type, "__getitem__", counted):
                audit = joint.audit_result(result, raw_path=directory)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(dict(reads), {key: len(rows) for key in raw})

    def test_final_weights_survive_selection_of_initial_checkpoint(self):
        args = spec.source.arguments(310001, preflight=True)
        dim = 6 + joint.scale_for(args).history_steps * 6
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=dim, lower_state_dim=dim, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, upper_learning_rate=3e-4, lower_learning_rate=3e-4))

        class ImmediatePool:
            def __init__(self, *, initializer, initargs, **kwargs):
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def map(self, function, jobs):
                return map(function, jobs)

        original = joint.worker_rollout
        selections = []

        def select_initial(job):
            batch, row = original(job)
            if not job[2] and job[3] is None:
                selections.append(row)
                row["charged_utility"] = 1000. - len(selections)
            return batch, row

        cell = {"selected_checkpoint_iteration": 0, "factual_row": {"decision_steps": list(range(0, args.horizon, 50))}}
        with tempfile.TemporaryDirectory() as directory, patch.object(joint, "load_controller", return_value=(source, cell, {})), \
                patch.object(joint, "ProcessPoolExecutor", ImmediatePool), patch.object(joint, "worker_rollout", select_initial), \
                patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            result = joint.train(310001, "learned_history", preflight=True, output=Path(directory) / "cell" / "result.json")
            selected = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            final = torch.load(result["final_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(selected["iteration"], 0)
            self.assertEqual(final["iteration"], 2)
            for level in ("upper_actor", "lower_actor", "promotion_actor"):
                self.assertTrue(any(not torch.equal(value, final["state_dict"][level][key])
                                    for key, value in selected["state_dict"][level].items()))

    def test_joint_gate_preserves_performance_and_capacity_control_failures(self):
        def cells(return_gain, current_gain=0.):
            results = []
            for root in spec.OPTIMIZER_ROOTS:
                for method in spec.METHODS:
                    row = {"episode_return": 100., "tracking_squared_error_integral": 2.,
                           "upper_inference_calls": 24., "charged_utility": 76.}
                    if method == "learned_history":
                        row.update(episode_return=100. + return_gain, tracking_squared_error_integral=1., upper_inference_calls=20., charged_utility=80. + return_gain)
                    if method == "learned_current":
                        row["charged_utility"] += current_gain
                    results.append({"root": root, "method": method, "evaluation_rows": [row]})
            return results
        self.assertEqual(joint.aggregate(cells(1))["status"], "stage35_development_gate_passed")
        failed = joint.aggregate(cells(-1))
        self.assertEqual(failed["status"], "stage35_development_gate_failed")
        self.assertTrue(failed["primary_endpoints"]["calls:fixed50"]["supported"])
        self.assertFalse(failed["primary_endpoints"]["return:fixed50"]["supported"])
        self.assertEqual(joint.aggregate(cells(1, current_gain=20))["status"], "stage35_development_gate_failed")
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            joint.aggregate(cells(1)[:-1])


if __name__ == "__main__":
    unittest.main()
from collections import Counter
from pathlib import Path
import tempfile
