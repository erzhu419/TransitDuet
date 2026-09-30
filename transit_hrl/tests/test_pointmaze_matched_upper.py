import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_matched_upper as experiment
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_matched_upper_stage57_spec as spec
from scripts.submit_pointmaze_matched_upper_stage57_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class MatchedUpperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.predictor, _ = learned.forecast.fit_forecaster(spec.arguments(310001, preflight=True),
            spec.SOURCE_SPEC.seed_roles(310001, preflight=True)["fitting"])

    def model(self):
        torch.manual_seed(57)
        return learned.make_model(FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390,
            lower_state_dim=390, upper_action_dim=2, lower_action_dim=2, hidden_dim=8,
            lower_cost_critic=False, lower_value_state_dim=392, epochs=1, minibatch_size=128)))

    def test_training_executes_zero_control_but_keeps_native_proposed_batches(self):
        model = self.model()
        experiment.init_worker(model.config, spec.arguments(310001, preflight=True))
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for arm in spec.TRAIN_POLICIES:
                with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                        patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                    batch, row = experiment.worker_rollout((joint.inference_weights(model), 123, arm, 50,
                        "train", "training", self.predictor, str(Path(directory) / f"{arm}.npz")))
                outputs.append((batch, row))
                self.assertGreater(row["proposed_action_rms"], 0)
                self.assertEqual(row["lower_training_credit"]["mode"], "task_option")
                with np.load(Path(directory) / f"{arm}.npz") as file:
                    raw = {k: file[k] for k in file.files}
                np.testing.assert_array_equal(batch.upper.action, raw["upper_proposed_action"])
                np.testing.assert_array_equal(batch.lower.reward, raw["reward"].astype(np.float32))
                for index, start in enumerate(row["decision_steps"]):
                    reward = raw["reward"][start:start + 50].copy()
                    reward[0] -= joint.spec.CALL_COST
                    self.assertAlmostEqual(batch.upper.reward[index], reward @ (model.config.gamma ** np.arange(50)), places=4)
            self.assertEqual(outputs[0][1]["initial_upper_action"], outputs[1][1]["initial_upper_action"])
            self.assertEqual(outputs[0][1]["executed_plan_delta_squared_sum"], 0)
            self.assertGreater(outputs[1][1]["executed_plan_delta_squared_sum"], 0)
            self.assertFalse(np.array_equal(outputs[0][0].lower.state, outputs[1][0].lower.state))

    def test_matched_pipeline_actor_freeze_updates_and_cost_mutations(self):
        model = self.model()
        original = copy.deepcopy(joint.inference_weights(model))
        source = {"config": json.loads(json.dumps(model.config.__dict__)), "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment, "load_source", return_value=({p: copy.deepcopy(model) for p in ("50", "100")}, self.predictor, source)), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                result = experiment.train(310001, preflight=True, output=Path(directory) / "run/result.json")
                summary = experiment.aggregate([result], preflight=True)
        self.assertEqual(summary["status"], "preflight_passed")
        self.assertEqual(sum(c["primitive_steps"] for c in summary["method_cost"].values()), 16800)
        self.assertEqual(summary["native_trace_audits"], 56)
        self.assertEqual(summary["optimizer_steps"], {"upper_actor": 4, "upper_value": 12, "lower_actor": 40, "lower_value": 80})
        self.assertNotIn("primary_endpoints", summary)
        torch.testing.assert_close(joint.inference_weights(model), original, atol=0, rtol=0)
        for p in ("50", "100"):
            self.assertEqual(result["training"][p]["zero_train"]["parameter_changes"]["upper_actor"], 0)
            self.assertEqual(result["training"][p]["zero_train"]["parameter_changes"]["upper_value"], 0)
            self.assertGreater(result["training"][p]["joint_ppo"]["parameter_changes"]["upper_actor"], 0)
            for arm in spec.TRAIN_POLICIES:
                self.assertEqual(result["calibration"][p][arm]["parameter_changes"]["lower_actor"], 0)
                self.assertGreater(result["training"][p][arm]["parameter_changes"]["lower_actor"], 0)
        for mutation, message in (("plan", "zero execution"), ("upper", "network changes"),
                                  ("steps", "optimizer budget"), ("credit", "native option credit")):
            bad = copy.deepcopy(result)
            item = bad["training"]["50"]["zero_train"]
            if mutation == "plan":
                item["history"][0]["rows"][0]["executed_action_rms"] = .1
            elif mutation == "upper":
                item["parameter_changes"]["upper_actor"] = .1
            elif mutation == "steps":
                item["history"][0]["optimizer_steps"]["lower_actor_optimizer_steps"] += 1
            else:
                item["history"][0]["rows"][0]["lower_training_credit"]["mode"] = "intrinsic"
            with self.assertRaisesRegex(ValueError, message):
                experiment.aggregate([bad], preflight=True)

    def test_source_loads_clone_not_joint_and_normalizes_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            file = directory / "source/result.json"
            file.parent.mkdir()
            raw = directory / "source_raw"
            raw.mkdir()
            np.savez_compressed(raw / "forecaster.npz", **self.predictor)
            model, checkpoints = self.model(), {}
            for period in spec.PERIODS:
                checkpoint = raw / f"clone{period}.pt"
                torch.save({"protocol": spec.SOURCE_SPEC.EXPERIMENT_PROTOCOL, "root": 310001,
                    "period": period, "policy": "clone", "state_dict": model.state_dict()}, checkpoint)
                checkpoints[str(period)] = {"clone": str(checkpoint)}
            file.write_text(json.dumps({"status": "complete", "protocol": spec.SOURCE_SPEC.EXPERIMENT_PROTOCOL,
                "contract": spec.SOURCE_SPEC.contract(), "root": 310001, "preflight": True,
                "options": spec.SOURCE_SPEC.options(preflight=True), "seed_roles": spec.SOURCE_SPEC.seed_roles(310001, preflight=True),
                "checkpoints": checkpoints, "config": model.config.__dict__, "budget": spec.SOURCE_SPEC.budget(preflight=True)}))
            with patch.object(spec, "source_result", return_value=file):
                loaded, _, _ = experiment.load_source(310001, preflight=True)
                torch.testing.assert_close(joint.inference_weights(loaded["50"]), joint.inference_weights(model), atol=0, rtol=0)
                checkpoint = checkpoints["50"]["clone"]
                payload = torch.load(checkpoint, weights_only=False)
                payload["policy"] = "joint_ppo"
                torch.save(payload, checkpoint)
                with self.assertRaisesRegex(ValueError, "fixed final clone"):
                    experiment.load_source(310001, preflight=True)

    def test_fresh_roles_dynamic_placement_budgets_and_contrasts(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                old = {s for previous in (spec.previous, spec.SOURCE_SPEC, spec.SOURCE_SPEC.previous) for values in
                       previous.seed_roles(root, preflight=preflight).values() for s in values}
                for values in spec.seed_roles(root, preflight=preflight).values():
                    self.assertFalse(set(values) & (old | seen))
                    seen.update(values)
            task = task_specification("unit_stage57", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 16588800)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 13824)
        means = {str(p): {"deterministic": {k: {"episode_return": v} for k, v in zip(spec.POLICIES, (0., 2., 3.))}} for p in spec.PERIODS}
        self.assertEqual(list(spec.contrasts(means).values()), [1., 3., 2.] * 2)
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, (1., 2., 0., -1., 2., 0.)))} for _ in range(8)]
        endpoints = experiment.bootstrap(rows)
        self.assertEqual([endpoints[k]["effect"] for k in spec.ENDPOINTS], ["positive", "positive", "inconclusive", "negative", "positive", "inconclusive"])


if __name__ == "__main__":
    unittest.main()
