import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_fresh_decoder as experiment
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig, concat_hierarchical_batches
from scripts import pointmaze_fresh_decoder_stage97_spec as spec
from scripts.submit_pointmaze_fresh_decoder_stage97_scheduleurm import task_specification, qualification_task
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class FreshDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self, directory):
        root = 410011
        args = spec.source.arguments(root, preflight=False)
        args.horizon = 300
        torch.manual_seed(root)
        original = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, lower_value_state_dim=392, hidden_dim=8, lower_cost_critic=False))
        learned = experiment.curves.support.learned
        template, fitted = learned.make_model(original), predictor()
        source_file = directory / "replicate_410011/result.json"
        source_file.parent.mkdir(parents=True)
        raw = source_file.parent.with_name(source_file.parent.name + "_raw")
        raw.mkdir()
        roles = spec.source.seed_roles(root, preflight=False)
        cell = {"root": root, "seed_roles": roles, "groups": {}, "checkpoints": {}, "config": template.config.__dict__}
        learned.init_worker(original.config, template.config, args)
        bounds = (-2 * np.ones(2), 2 * np.ones(2))
        for period in spec.PERIODS:
            path = raw / str(period) / "teacher/labels/0/deterministic"
            path.mkdir(parents=True)
            with patch.object(learned.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(learned.joint, "pointmaze_goal_bounds", return_value=bounds):
                pairs = [learned.worker_rollout((learned.joint.inference_weights(template), seed, "teacher", period,
                    "labels", "deterministic", learned.feedback_gain(), fitted, str(path / f"episode_{seed}.npz"))) for seed in roles["labels"]]
            clone, result = learned.clone(template, concat_hierarchical_batches([batch for batch, _ in pairs]).lower,
                root=root, period=period, epochs=64, sham=False)
            checkpoint = raw / f"clone_{period}_final.pt"
            torch.save({"protocol": spec.source.EXPERIMENT_PROTOCOL, "root": root, "preflight": False, "period": period,
                "bc_epochs": 64, "frozen_std_and_upper": "passed", "config": clone.config.__dict__,
                "weights": learned.joint.inference_weights(clone)}, checkpoint)
            cell["checkpoints"][f"clone_{period}"] = str(checkpoint)
            cell["groups"][str(period)] = {"cloning": result}
        cell["forecaster"] = str(raw / "forecaster.npz")
        np.savez_compressed(cell["forecaster"], **fitted)
        source_file.write_text(json.dumps(cell))
        return args, source_file

    def test_fresh_probes_full_labels_realized_budget_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertEqual(roles["calibration_labels"], spec.source.seed_roles(root, preflight=False)["labels"])
                self.assertFalse(set(roles["native_probe"]) & seen)
                self.assertFalse(set(roles["native_probe"]) & set(sum(spec.source.seed_roles(root, preflight=False).values(), [])))
                seen.update(roles["native_probe"])
                task = task_specification("unit_stage97", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage97", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        self.assertEqual(spec.roots(preflight=True), (410011,))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (16, 4800))
        self.assertEqual((spec.budget(preflight=False)["native_episodes"] * 8, spec.budget(preflight=False)["native_steps"] * 8), (128, 153600))
        self.assertEqual(spec.budget(preflight=True)["label_state_rows"], 19200)

    def test_actual_BC_reconstruction_fixed_scale_probes_and_new_source_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args, source_file = self.source(directory)
            bounds = (-2 * np.ones(2), 2 * np.ones(2))
            events = []
            calibrator, worker = experiment.calibrate, experiment.worker_probe

            def calibrate(*a, **kw):
                value = calibrator(*a, **kw)
                events.append("calibrated")
                return value

            def probe(job):
                events.append("probe")
                return worker(job)

            with patch.object(spec.source, "arguments", return_value=args), \
                    patch.object(spec, "source_result", return_value=source_file), \
                    patch.object(experiment.teachers, "qualify"), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "calibrate", side_effect=calibrate), \
                    patch.object(experiment, "worker_probe", side_effect=probe), \
                    patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=bounds):
                result = experiment.run(410011, preflight=True, output=directory / "decoder/result.json")
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["performance_confirmation"], "not_tested")
                self.assertEqual(events[:3], ["calibrated", "calibrated", "probe"])
                self.assertEqual(result["cost"]["native_steps"], 4800)
                evaluations = sum(len(g["calibration"]["solver_trace"]) for g in result["groups"].values())
                self.assertEqual(result["cost"]["constraint_response_evaluations"], evaluations)
                self.assertEqual(result["cost"]["offline_actor_rows"], (4 + evaluations) * 8 * 300)
                self.assertTrue((directory / "decoder/completion/ready.json").is_file())
                for group in result["groups"].values():
                    c = group["calibration"]
                    self.assertLessEqual(c["responses"]["bounded"]["command_change_rms"], c["target_rms"])
                    self.assertEqual(c["envelope"]["rows"], 2400)
                    self.assertEqual(c["data_role"], "new_cohort_BC_labels_only")
                bad = copy.deepcopy(result)
                bad["cost"]["offline_actor_forward_batches"] -= 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(result)
                bad["groups"]["50"]["calibration"]["target_rms"] *= 2
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(result)
                bad["groups"]["100"]["evaluation"]["bounded"][0]["lower_seed"] += 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(result)
                for group in bad["groups"].values():
                    for row in group["evaluation"]["bounded"]: row["episode_return"] = -1000000.
                experiment.qualify(bad, preflight=True)
                payload_path = json.loads(source_file.read_text())["checkpoints"]["clone_50"]
                payload = torch.load(payload_path, weights_only=False)
                payload["root"] = 310011
                torch.save(payload, payload_path)
                with self.assertRaisesRegex(ValueError, "fixed-final new cohort"): experiment.load_source(410011)

    def test_entire_cohort_is_required_and_nonlinear_rule_retained(self):
        with self.assertRaisesRegex(ValueError, "cohort incomplete"):
            experiment.aggregate([{"root": 410011}], preflight=False)
        trace = experiment.bounded.contract_response(.5, .1,
            lambda alpha: {"command_change_rms": .15 if alpha == .25 else .08}, {"command_change_rms": .2})
        self.assertEqual([r["alpha"] for r in trace], [.5, .25, .125])


if __name__ == "__main__":
    unittest.main()
