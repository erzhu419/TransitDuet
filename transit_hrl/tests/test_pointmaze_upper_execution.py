import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_upper_execution as experiment
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.experiments import pointmaze_forecast_tracking as forecast
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_upper_execution_stage56_spec as spec
from scripts.submit_pointmaze_upper_execution_stage56_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class UpperExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.predictor, _ = forecast.fit_forecaster(spec.arguments(310001, preflight=True),
            spec.previous.seed_roles(310001, preflight=True)["fitting"])

    def model(self):
        torch.manual_seed(56)
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False,
            lower_value_state_dim=392, epochs=1, minibatch_size=128))
        model = learned.make_model(source)
        with torch.no_grad():
            model.upper_actor.net[-1].bias.copy_(torch.tensor([.12, -.07, .05, -.08]))
        return model

    def source_fixture(self, directory):
        path = directory / "source/result.json"
        path.parent.mkdir()
        raw = directory / "source_raw"
        raw.mkdir()
        np.savez_compressed(raw / "forecaster.npz", **self.predictor)
        model, checkpoints = self.model(), {}
        for period in spec.PERIODS:
            file = raw / f"joint_{period}.pt"
            torch.save({"protocol": spec.previous.EXPERIMENT_PROTOCOL, "root": 310001,
                "period": period, "policy": "joint_ppo", "state_dict": model.state_dict()}, file)
            checkpoints[str(period)] = {"joint_ppo": str(file)}
        path.write_text(json.dumps({"status": "complete", "protocol": spec.previous.EXPERIMENT_PROTOCOL,
            "contract": spec.previous.contract(), "root": 310001, "preflight": True,
            "options": spec.previous.options(preflight=True), "seed_roles": spec.previous.seed_roles(310001, preflight=True),
            "checkpoints": checkpoints, "config": model.config.__dict__, "budget": spec.previous.budget(preflight=True)}))
        return path

    def test_zero_executes_base_without_zeroing_proposed_action(self):
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.001, .002]
        kwargs = dict(observation=SimpleNamespace(task_measurement=frames[-1]),
            history=SimpleNamespace(history=frames.reshape(-1)), step=63, world_low=-2 * np.ones(2), world_high=2 * np.ones(2))
        action = np.array([.2, -.1, .1, -.2], dtype=np.float32)
        normal, zero = [experiment.ExecutedPlan(self.predictor, 100, .5, p) for p in spec.POLICIES]
        normal.decode(action=action, **kwargs)
        zero.decode(action=action, **kwargs)
        np.testing.assert_array_equal(normal.proposed_actions, zero.proposed_actions)
        np.testing.assert_array_equal(normal.actions, [action])
        np.testing.assert_array_equal(zero.actions, [np.zeros(4)])
        np.testing.assert_array_equal(zero.points, forecast.plan_points(frames[:, :2], "ridge_velocity", self.predictor, 100, zero.bounds))
        self.assertEqual(zero.executed_delta_squared_sum, 0.)
        self.assertGreater(normal.executed_delta_squared_sum, 0.)
        np.testing.assert_array_equal(normal.points[0], zero.points[0])
        saved = normal.points.copy()
        kwargs["history"].history[:] = 999
        np.testing.assert_array_equal(normal(age=25), saved[25])

    def test_paired_rollouts_native_audit_and_trace_mutation_rejection(self):
        model, args = self.model(), spec.arguments(310001, preflight=True)
        weights = joint.inference_weights(model)
        experiment.init_worker(model.config, args)
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for policy in spec.POLICIES:
                path = Path(directory) / f"{policy}.npz"
                with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                        patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                    row = experiment.worker_rollout((weights, 123, policy, 50, "lower_sampled", self.predictor, str(path)))
                rows.append(row)
                with np.load(path) as archive:
                    raw = {k: archive[k] for k in archive.files}
                self.assertEqual(row["frozen_network_check"], "passed")
                self.assertEqual((row["upper_inference_calls"], row["lower_inference_calls"]), (6, 300))
                self.assertAlmostEqual(row["episode_return"], raw["reward"].sum())
                experiment.audit_intervention(raw, row, predictor=self.predictor, period=50,
                    scale=args.maximum_subgoal_delta, bounds=(-2, 2))
                bad = copy.deepcopy(raw)
                bad["upper_plan_action"][1, 0] += .2
                with self.assertRaises(AssertionError):
                    experiment.audit_intervention(bad, row, predictor=self.predictor, period=50,
                        scale=args.maximum_subgoal_delta, bounds=(-2, 2))
            self.assertEqual(rows[0]["initial_upper_action"], rows[1]["initial_upper_action"])
            self.assertEqual(rows[0]["lower_seed"], rows[1]["lower_seed"])
            self.assertGreater(rows[1]["proposed_action_rms"], 0)
            self.assertEqual(rows[1]["executed_action_rms"], 0)
        torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)

    def test_fixed_source_pipeline_costs_and_reachable_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = self.source_fixture(directory)
            with patch.object(spec, "source_result", return_value=source), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                result = experiment.evaluate(310001, preflight=True, output=directory / "run/result.json")
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["method_cost"]["primitive_steps"], 4800)
                self.assertEqual(summary["native_trace_audits"], 16)
                self.assertEqual(summary["optimizer_steps"], 0)
                self.assertEqual(summary["new_forecaster_fits"], 0)
                self.assertNotIn("primary_endpoints", summary)
                for mutation, message in (("zero", "nonzero plan"), ("frozen", "learned execution"),
                                          ("pair", "paired initial"), ("cost", "budget")):
                    bad = copy.deepcopy(result)
                    row = bad["evaluation_rows"]["50"]["zero_residual"]["deterministic"][0]
                    if mutation == "zero":
                        row["executed_action_rms"] = .1
                    elif mutation == "frozen":
                        row["frozen_network_check"] = "changed"
                    elif mutation == "pair":
                        row["initial_upper_action"][0] += .1
                    else:
                        bad["inference_counts"]["upper_inference_calls"] += 1
                    with self.assertRaisesRegex(ValueError, message):
                        experiment.aggregate([bad], preflight=True)
                cell = json.loads(source.read_text())
                file = cell["checkpoints"]["50"]["joint_ppo"]
                payload = torch.load(file, weights_only=False)
                payload["policy"] = "clone"
                torch.save(payload, file)
                with self.assertRaisesRegex(ValueError, "fixed final joint"):
                    experiment.load_source(310001, preflight=True)

    def test_disjoint_paths_dynamic_placement_and_simultaneous_endpoints(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                old = {s for previous in (spec.previous, spec.previous.previous) for values in
                    previous.seed_roles(root, preflight=preflight).values() for s in values}
                seeds = set(spec.seed_roles(root, preflight=preflight)["evaluation"])
                self.assertFalse(seeds & (seen | old))
                seen.update(seeds)
            task = task_specification("unit_stage56", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 1228800)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 1024)
        means = {str(p): {"deterministic": {"normal": {"episode_return": 5.}, "zero_residual": {"episode_return": 3.}}} for p in spec.PERIODS}
        self.assertEqual(list(spec.contrasts(means).values()), [2., 2.])
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, (1., -1.)))} for _ in range(8)]
        result = experiment.bootstrap(rows)
        self.assertEqual(result[spec.ENDPOINTS[0]]["effect"], "positive")
        self.assertEqual(result[spec.ENDPOINTS[1]]["effect"], "negative")


if __name__ == "__main__":
    unittest.main()
