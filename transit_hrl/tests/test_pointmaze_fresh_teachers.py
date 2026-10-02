import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_fresh_teachers as experiment
from freq_hrl.rl.goal_conditioned_actor_critic import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig
from scripts import pointmaze_fresh_teachers_stage96_spec as spec
from scripts import pointmaze_staged_upper_confirmation_stage95_spec as old
from scripts.submit_pointmaze_fresh_teachers_stage96_scheduleurm import task_specification, qualification_task
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_call_weighted import seeds as mc_seeds


class NativeFixture(DenseTask):
    def __init__(self):
        super().__init__()
        self.driver = SimpleNamespace(regime_change_steps=[])

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, {**info, "tracking_success": 0., "executed_action": np.asarray(action)}

    def diagnostics(self):
        return {}


class FreshTeacherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self):
        torch.manual_seed(410001)
        return GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(upper_state_dim=390, lower_state_dim=390,
            goal_dim=2, action_dim=2, hidden_dim=8, epochs=4, minibatch_size=1024, init_log_std=-.7))

    def test_new_cohort_and_all_build_roles_disjoint_and_budget_dynamic(self):
        seeds = []
        self.assertFalse(set(spec.OPTIMIZER_ROOTS) & set(old.roots(preflight=False)))
        self.assertFalse(set(spec.PREFLIGHT_ROOTS) & set(spec.OPTIMIZER_ROOTS))
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds.extend(sum(roles.values(), []))
                task = task_specification("unit_stage96", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage96", preflight=preflight)
            self.assertIn(spec.ANALYZER_SCRIPT, q["cmd"])
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        self.assertEqual(len(seeds), len(set(seeds)))
        old_seeds = []
        for root in old.roots(preflight=False):
            old_seeds.extend(mc_seeds(old.seed_roles(root, preflight=False)))
        self.assertFalse(set(seeds) & set(old_seeds))
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"] * 8, b["native_steps"] * 8), (25984, 31180800))
        self.assertEqual((b["bc_optimizer_steps"], b["controller_lower_actor_steps"], b["warmup_actor_steps"]), (1280, 15360, 0))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (12, 3600))

    def pipeline(self, directory):
        model = self.source()
        args = spec.arguments(410001, preflight=True)
        args.reference_hidden_dim = 8
        bounds = (-2 * np.ones(2), 2 * np.ones(2))
        with patch.object(spec, "arguments", return_value=args), \
                patch.object(experiment, "make_controller", return_value=(model, {"reference_parameter_budget": model.trainable_parameter_count})), \
                patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                patch.object(experiment.task_core, "_make_task", side_effect=lambda **kw: NativeFixture()), \
                patch.object(experiment.task_core, "pointmaze_goal_bounds", return_value=bounds), \
                patch.object(experiment.task_core, "_causal_distinguishability_diagnostics", return_value={}), \
                patch.object(experiment.task_core, "_event_diagnostics", return_value={}), \
                patch.object(experiment.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                patch.object(experiment.joint, "pointmaze_goal_bounds", return_value=bounds):
            return experiment.run(410001, preflight=True, output=directory / "cell/result.json")

    def test_full_prerequisite_pipeline_uses_final_native_weights_no_historical_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.object(experiment.torch, "load", side_effect=AssertionError("historical checkpoint read")):
                result = self.pipeline(directory)
            self.assertEqual(experiment.aggregate([result], preflight=True)["status"], "preflight_passed")
            self.assertEqual(result["native_counts"]["steps"], 3600)
            self.assertEqual(result["records"]["controller"]["final_iteration"], 2)
            self.assertTrue(all(v > 0 for v in result["records"]["controller"]["actor_change_norms"].values()))
            self.assertEqual(result["historical_artifact_loads"], 0)
            self.assertTrue((directory / "cell/completion/ready.json").is_file())
            clock = torch.load(result["checkpoints"]["clock"], weights_only=False)
            controller = torch.load(result["checkpoints"]["controller"], weights_only=False)
            self.assertEqual(controller["checkpoint_selection"], "fixed_final")
            for period in spec.PERIODS:
                payload = torch.load(result["checkpoints"][f"clone_{period}"], weights_only=False)
                self.assertEqual((payload["protocol"], payload["root"], payload["period"]), (spec.EXPERIMENT_PROTOCOL, 410001, period))
                weights = payload["weights"]
                torch.testing.assert_close(weights["lower_actor"]["log_std"], clock["weights"]["lower_actor"]["log_std"], atol=0, rtol=0)
                self.assertTrue(torch.all(weights["upper_actor"]["net.4.weight"] == 0))
                self.assertGreater(result["groups"][str(period)]["cloning"]["steps"], 0)
                for seed in result["seed_roles"]["labels"]:
                    path = Path(result["groups"][str(period)]["label_archive_directory"]) / f"episode_{seed}.npz"
                    with np.load(path) as archive:
                        self.assertEqual(archive["lower_actor_context"].shape, (300, 2))
                        self.assertTrue(np.all(archive["upper_plan_action"] == 0))
            bad = copy.deepcopy(result)
            bad["records"]["controller"]["final_iteration"] = 1
            with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
            bad = copy.deepcopy(result)
            bad["records"]["warmup"]["history"][0]["optimizer_steps"]["lower_actor_optimizer_steps"] = 1
            with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
            bad = copy.deepcopy(result)
            bad["groups"]["50"]["label_rows"][0]["seed"] -= 1
            with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
            bad = copy.deepcopy(result)
            bad["historical_artifact_loads"] = 1
            with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
            bad = copy.deepcopy(result)
            for rows in bad["records"]["controller"]["diagnostics"].values():
                for row in rows: row["episode_return"] = -1000000.
            self.assertEqual(experiment.aggregate([bad], preflight=True)["performance_confirmation"], "not_tested")
            Path(result["groups"]["100"]["label_archive_directory"], f"episode_{result['seed_roles']['labels'][0]}.npz").unlink()
            with self.assertRaisesRegex(ValueError, "label archive missing"): experiment.aggregate([result], preflight=True)

    def test_analyzer_requires_entire_new_cohort(self):
        with self.assertRaisesRegex(ValueError, "cohort incomplete"):
            experiment.aggregate([{"root": spec.OPTIMIZER_ROOTS[0]}], preflight=False)


if __name__ == "__main__":
    unittest.main()
