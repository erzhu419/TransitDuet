import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_credit_diagnostics as experiment
from freq_hrl.experiments import pointmaze_matched_upper as native
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_credit_diagnostics_stage62_spec as spec
from scripts.submit_pointmaze_credit_diagnostics_stage62_scheduleurm import task_specification
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask


class CreditDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_boundary_bootstrap_trace_and_episode_semantics(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=1, lower_state_dim=1,
            upper_action_dim=1, lower_action_dim=1, hidden_dim=4, lower_cost_critic=False, gamma=.9, gae_lambda=1.))
        data = {"reward": np.ones(4, dtype=np.float32), "upper_reward": np.full(2, 1.9, dtype=np.float32),
                "upper_state": np.zeros((2, 1)), "lower_value_state": np.zeros((4, 1))}
        with patch.object(experiment, "predict", side_effect=[(np.full(4, 10., dtype=np.float32), 1),
                (np.full(2, 10., dtype=np.float32), 1)]):
            terms = experiment.episode_terms(model, data, 2)
        np.testing.assert_allclose(terms["option_mc"], [1.9, 1., 1.9, 1.])
        np.testing.assert_allclose(terms["episode_mc"], [3.439, 2.71, 1.9, 1.])
        np.testing.assert_allclose(terms["upper_mc"], [3.439, 1.9])
        np.testing.assert_allclose(terms["option_adv"], [-8.1, -9., -8.1, -9.], atol=1e-6)
        np.testing.assert_allclose(terms["bootstrap_adv"], [0., 0., -8.1, -9.], atol=1e-6)
        np.testing.assert_allclose(terms["episode_adv"], [-6.561, -7.29, -8.1, -9.], atol=1e-6)
        np.testing.assert_allclose(terms["boundary_bootstrap"], [9.])
        summary = experiment.summarize([terms])
        self.assertEqual(summary["boundary"]["count"], 1)
        self.assertEqual(summary["boundary"]["option_terminal_td_mean"], -9.)
        self.assertEqual(summary["boundary"]["continuing_td_mean"], 0.)

    def test_native_feature_and_macro_reward_reconstruction_exact(self):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        helper = archive_data.UpdateDiagnosticsTest()
        model = helper.model()
        args = spec.arguments(310001, preflight=True)
        before = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            for period in spec.PERIODS:
                for arm in spec.TRAIN_POLICIES:
                    batch, _, path = helper.archived_episode(model, directory, arm=arm, period=period)
                    with np.load(path) as archive:
                        raw = {k: archive[k] for k in archive.files}
                    data = experiment.features(raw, args, period, model.config.gamma)
                    np.testing.assert_array_equal(data["upper_state"], batch.upper.state)
                    np.testing.assert_array_equal(data["lower_value_state"], batch.lower.value_state)
                    np.testing.assert_array_equal(data["upper_reward"], batch.upper.reward)
                    np.testing.assert_array_equal(data["reward"], batch.lower.reward)
                    terms = experiment.episode_terms(model, data, period)
                    self.assertEqual(terms["cost"]["lower_value_forward_batches"], 1)
                    self.assertEqual(len(terms["boundary_value"]), args.horizon // period - 1)
                    bad = dict(raw)
                    bad["decision_steps"] = raw["decision_steps"] + 1
                    with self.assertRaises(AssertionError):
                        experiment.features(bad, args, period, model.config.gamma)
        actual, expected = model.state_dict(), dict(before)
        self.assertEqual(actual.pop("config"), expected.pop("config"))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_metrics_and_frozen_cost_dynamic_pool(self):
        self.assertEqual(experiment.value_metrics([1., 2., 3.], [1., 2., 3.])["explained_variance"], 1.)
        self.assertIsNone(experiment.value_metrics([1., 2.], [1., 1.])["explained_variance"])
        self.assertEqual(experiment.alignment([1., 2., 3.], [2., 4., 6.])["normalized_sign_disagreement"], 0.)
        self.assertAlmostEqual(experiment.alignment([1., 2., 3.], [-1., -2., -3.])["correlation"], -1.)
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 768)
        self.assertEqual(8 * b["feature_lower_rows"], 921600)
        self.assertEqual(8 * b["feature_upper_rows"], 13824)
        self.assertEqual(8 * b["lower_value_forward_batches"], 2304)
        self.assertEqual(8 * b["upper_value_forward_batches"], 768)
        self.assertEqual(8 * b["gae_calls"], 3072)
        self.assertEqual(8 * b["mc_return_calls"], 2304)
        for preflight in (True, False):
            task = task_specification("unit_stage62", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (2, 4096))

    def test_archive_pipeline_frozen_weights_counts_seeds_and_qualification(self):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        helper = archive_data.UpdateDiagnosticsTest()
        model, args = helper.model(), spec.arguments(310001, preflight=True)
        native.init_worker(model.config, args)
        roles = spec.seed_roles(310001, preflight=True)
        common = {"status": "complete", "root": 310001, "preflight": True}
        source = {**common, "protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(),
                  "checkpoints": {}, "evaluation_rows": {}}
        training = {**common, "protocol": spec.source.deployment.EXPERIMENT_PROTOCOL,
                    "contract": spec.source.deployment.contract(), "training": {}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            training_file, source_file = directory / "stage57/result.json", directory / "stage61/result.json"
            training_raw = training_file.parent.with_name(training_file.parent.name + "_raw")
            source_raw = source_file.parent.with_name(source_file.parent.name + "_raw")
            with patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                for period in spec.PERIODS:
                    p = str(period)
                    source["checkpoints"][p], source["evaluation_rows"][p], training["training"][p] = {}, {}, {}
                    for arm in spec.TRAIN_POLICIES:
                        path = source_raw / p / arm / spec.TREATMENT
                        path.mkdir(parents=True)
                        checkpoint = path / "policy.pt"
                        torch.save({"protocol": spec.source.EXPERIMENT_PROTOCOL, "root": 310001, "period": period,
                            "arm": arm, "treatment": spec.TREATMENT, "state_dict": model.state_dict()}, checkpoint)
                        source["checkpoints"][p][arm] = {spec.TREATMENT: str(checkpoint)}
                        for split in spec.SPLITS:
                            raw_dir = training_raw / p / arm / "train/1/training" if split == spec.SPLITS[0] else path
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            rows = [native.worker_rollout((joint.inference_weights(model), seed, arm, period,
                                "train" if split == spec.SPLITS[0] else "eval",
                                "training" if split == spec.SPLITS[0] else "deterministic", helper.predictor,
                                str(raw_dir / f"episode_{seed}.npz")))[1] for seed in roles[split]]
                            if split == spec.SPLITS[0]:
                                training["training"][p][arm] = {"history": [{"iteration": 1, "rows": rows}]}
                            else:
                                source["evaluation_rows"][p][arm] = {spec.TREATMENT: rows}
            for path, c in ((training_file, training), (source_file, source)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(c))
            with patch.object(spec, "source_result", return_value=source_file), \
                    patch.object(spec.training_source, "source_result", return_value=training_file), \
                    patch.object(joint, "_make_task", side_effect=AssertionError("diagnosis must not sample")), \
                    patch.object(FrequencySeparatedActorCriticPPO, "_update_level", side_effect=AssertionError("diagnosis must not train")), \
                    patch.object(FrequencySeparatedActorCriticPPO, "act_lower", side_effect=AssertionError("diagnosis must not call actor")), \
                    patch.object(FrequencySeparatedActorCriticPPO, "act_upper", side_effect=AssertionError("diagnosis must not call actor")):
                result = experiment.diagnose(310001, preflight=True, output=directory / "stage62/result.json")
                summary = experiment.aggregate([result], preflight=True)
            self.assertEqual(summary["status"], "preflight_passed")
            self.assertEqual(summary["cost"], spec.budget(preflight=True))
            self.assertEqual(summary["performance_claim"], "none_diagnostic_only")
            for mutation in ("cost", "state", "seed", "boundary"):
                bad = copy.deepcopy(result)
                if mutation == "cost":
                    bad["cost"]["native_steps"] = 1
                elif mutation == "state":
                    bad["frozen_state_checks"]["50"]["joint_ppo"] = "failed"
                elif mutation == "seed":
                    bad["groups"]["50"]["joint_ppo"][spec.SPLITS[1]]["seeds"] = [1, 2]
                else:
                    bad["groups"]["50"]["joint_ppo"][spec.SPLITS[1]]["metrics"]["boundary"]["count"] += 1
                with self.assertRaises(ValueError):
                    experiment.qualify(bad, preflight=True)
            with self.assertRaises(ValueError):
                experiment.aggregate([result, result], preflight=True)


if __name__ == "__main__":
    unittest.main()
