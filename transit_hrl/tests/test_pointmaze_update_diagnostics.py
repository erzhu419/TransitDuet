import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_update_diagnostics as experiment
from freq_hrl.experiments import pointmaze_matched_upper as previous
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_update_diagnostics_stage58_spec as spec
from scripts.submit_pointmaze_update_diagnostics_stage58_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class UpdateDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.predictor, _ = learned.forecast.fit_forecaster(spec.previous.arguments(310001, preflight=True),
            spec.previous.SOURCE_SPEC.seed_roles(310001, preflight=True)["fitting"])

    def model(self):
        torch.manual_seed(58)
        return learned.make_model(FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390,
            lower_state_dim=390, upper_action_dim=2, lower_action_dim=2, hidden_dim=8,
            lower_cost_critic=False, lower_value_state_dim=392, epochs=1, minibatch_size=128)))

    def archived_episode(self, model, directory, *, arm="joint_ppo", period=50):
        previous.init_worker(model.config, spec.previous.arguments(310001, preflight=True))
        path = Path(directory) / f"{arm}{period}.npz"
        seed = spec.previous.seed_roles(310001, preflight=True)["warmup"][0]
        with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
            batch, row = previous.worker_rollout((joint.inference_weights(model), seed, arm, period,
                "warmup", "training", self.predictor, str(path)))
        return batch, row, path

    def test_reconstructed_batches_match_native_batches_and_detect_wrong_actions(self):
        model = self.model()
        experiment.init_worker(model.config, spec.previous.arguments(310001, preflight=True))
        with tempfile.TemporaryDirectory() as directory:
            for period in spec.PERIODS:
                for arm in spec.TRAIN_POLICIES:
                    native, row, path = self.archived_episode(model, directory, arm=arm, period=period)
                    replayed, result = experiment.worker_reconstruct((joint.inference_weights(model), str(path), row["seed"], period))
                    self.assertEqual(result["action_check"], "passed")
                    self.assertEqual(result["episode_return"], row["episode_return"])
                    for level in ("upper", "lower"):
                        for key, value in vars(getattr(native, level)).items():
                            other = getattr(getattr(replayed, level), key)
                            if value is None:
                                self.assertIsNone(other)
                            else:
                                np.testing.assert_array_equal(value, other)
            with np.load(path) as archive:
                raw = {k: archive[k] for k in archive.files}
            raw["upper_proposed_action"][0, 0] += .1
            with self.assertRaises(AssertionError):
                experiment.reconstruct(model, spec.previous.arguments(310001, preflight=True), raw,
                    seed=row["seed"], period=period)

    def test_diagnostics_preserve_ppo_networks_and_adam_exactly(self):
        initial = self.model()
        with tempfile.TemporaryDirectory() as directory:
            batch, _, _ = self.archived_episode(initial, directory)
        for phase in ("warmup", "train"):
            for level in ("upper", "lower"):
                plain, instrumented = copy.deepcopy(initial), copy.deepcopy(initial)
                np.random.seed(spec.previous.shuffle_seed(310001, 50, 1, phase=phase, level=level))
                expected = plain._update_level(level=level, batch=getattr(batch, level),
                    actor=getattr(plain, level + "_actor"), value_net=getattr(plain, level + "_value"),
                    actor_optimizer=getattr(plain, level + "_actor_optimizer"),
                    value_optimizer=getattr(plain, level + "_value_optimizer"), actor_updates_enabled=phase == "train")
                observed = experiment.observed_update(instrumented, batch, level=level, phase=phase,
                    root=310001, period=50, iteration=1, episode_count=1)
                torch.testing.assert_close(plain.state_dict(), instrumented.state_dict(), atol=0, rtol=0)
                self.assertEqual(observed["ppo_metrics"], expected)
                self.assertGreaterEqual(observed["kl_mean"], 0)
                self.assertLessEqual(observed["clip_fraction"], 1)
                self.assertGreaterEqual(observed["clip_fraction"], 0)
                self.assertGreaterEqual(observed["value_after"]["mse"], 0)
                if phase == "warmup":
                    self.assertEqual(observed["kl_max"], 0)
                    self.assertEqual(observed["std_before"], observed["std_after"])
                else:
                    self.assertGreater(observed["kl_mean"], 0)

    def test_end_to_end_replay_matches_final_parameters_and_all_optimizer_steps(self):
        model = self.model()
        source = {"config": json.loads(json.dumps(model.config.__dict__)),
            "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "stage57/result.json"
            with patch.object(previous, "load_source", side_effect=lambda *args, **kwargs:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, self.predictor, source)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(spec, "source_result", return_value=file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                previous.train(310001, preflight=True, output=file)
                with patch.object(joint, "_make_task", side_effect=AssertionError("replay must never collect an environment step")):
                    result = experiment.replay(310001, preflight=True, output=Path(directory) / "stage58/result.json")
                summary = experiment.aggregate([result], preflight=True)
        self.assertEqual(summary["status"], "preflight_passed")
        self.assertEqual(summary["cost"], spec.budget(preflight=True))
        self.assertEqual(summary["cost"]["native_steps"], 0)
        self.assertEqual(summary["replayed_optimizer_steps"],
            {"upper_actor": 4, "upper_value": 12, "lower_actor": 40, "lower_value": 80})
        for mutation in ("count", "identity", "actor", "steps"):
            bad = copy.deepcopy(result)
            if mutation == "count":
                bad["cost"]["archive_episodes"] += 1
            elif mutation == "identity":
                bad["final_state_checks"]["50"]["joint_ppo"] = "failed"
            elif mutation == "actor":
                bad["histories"]["50"]["warmup"]["zero_train"][0]["levels"][0]["kl_max"] = .1
            else:
                bad["histories"]["50"]["train"]["zero_train"][0]["levels"][0]["optimizer_steps"]["lower_actor_optimizer_steps"] += 1
            with self.assertRaises(ValueError):
                experiment.qualify(bad, preflight=True)

    def test_value_metrics_and_replay_budget_dynamic_placement(self):
        self.assertEqual(experiment.value_terms([1., 2., 3.], [1., 2., 3.]), {"mse": 0., "explained_variance": 1.})
        self.assertIsNone(experiment.value_terms([1., 2.], [1., 1.])["explained_variance"])
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 12288)
        self.assertEqual(8 * b["reconstructed_lower_calls"], 14745600)
        self.assertEqual(8 * b["reconstructed_upper_calls"], 221184)
        self.assertEqual(8 * b["diagnostic_updates"], 2560)
        for preflight in (True, False):
            task = task_specification("unit_stage58", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])


if __name__ == "__main__":
    unittest.main()
