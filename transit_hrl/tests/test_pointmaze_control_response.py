import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_control_response as experiment
from scripts import pointmaze_control_response_stage109_spec as spec
from scripts.submit_pointmaze_control_response_stage109_scheduleurm import task_specification, qualification_task
import test_pointmaze_crossed_advice as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class Float32Task(DenseTask):
    def observation(self):
        obs = super().observation()
        for name in ("physical", "achieved_goal", "target", "task_measurement"):
            setattr(obs, name, getattr(obs, name).astype(np.float32))
        obs.target_error = obs.target-obs.achieved_goal
        return obs


class ControlResponseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_decoder_matches_production_clip_blend_and_zero_identity(self):
        models, pred, _, cal = fixture.CrossedAdviceTest().sources()
        bounds = (-.4*np.ones(2), .4*np.ones(2))
        history = SimpleNamespace(history=np.tile([.3, -.3, 0., 0., 0., 0.], 64).astype(np.float32))
        for alpha in (.02, 1.):
            plan = experiment.source.native.curves.CalibratedPlan(pred, 50, 2., alpha, cal["50"]["envelope"])
            for action in (np.zeros(4), np.array([.3, -1., .7, 3.])):
                plan.decode(action=action, observation=None, history=history, step=50, world_low=bounds[0], world_high=bounds[1])
                decoded = experiment.decode_points(plan.base_points, plan, action, bounds, alpha)
                np.testing.assert_array_equal(decoded, plan.points)
                if not action.any():
                    np.testing.assert_array_equal(decoded, plan.base_points)

    def test_common_states_nonzero_response_replay_and_unchanged_models(self):
        models, pred, _, cal = fixture.CrossedAdviceTest().sources()
        model = models["50"]["learned_hint"]
        before = copy.deepcopy(model.state_dict())
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            experiment.init_worker(model.config, args)
            row = experiment.worker_episode((experiment.source.native.joint.inference_weights(model),
                109095001, 50, pred, .02, cal["50"]["envelope"]))
        self.assertEqual(row["trajectory"]["upper_calls"], 0)
        self.assertEqual(row["probe_states"], 20)
        self.assertEqual(row["responses"]["zero"]["command_delta_rms"], 0.)
        self.assertGreater(row["responses"]["full_sample"]["command_delta_rms"], 0.)
        self.assertGreater(row["responses"]["full_secant"]["command_Jacobian_frobenius_rms"], 0.)
        experiment.source.native.curves.support.assert_frozen(model, before)

    def test_response_KL_and_secant_have_known_linear_solution(self):
        class Actor:
            def distribution(self, x):
                return torch.distributions.Normal(x[:, 392:394]*2, torch.full((len(x), 2), .5))
        states = {k: np.zeros((10, 396), dtype=np.float32) for k in spec.PROBES}
        for s in spec.SCALES:
            for i in range(4):
                if i < 2:
                    states[f"{s}_axis{i}_plus"][:, 392+i] = spec.EPSILON
                    states[f"{s}_axis{i}_minus"][:, 392+i] = -spec.EPSILON
        states["full_mean"][:, 392] = .25
        result = experiment.responses(Actor(), states)
        self.assertEqual(result["full_mean"]["same_covariance_KL_mean"], .5)
        expected = np.tanh(.5)/spec.EPSILON
        self.assertAlmostEqual(result["full_secant"]["largest_singular_mean"], expected, places=7)
        self.assertAlmostEqual(result["full_secant"]["smallest_singular_mean"], expected, places=7)

    def test_reduced_run_cost_source_freeze_and_no_new_checkpoints(self):
        sources = fixture.CrossedAdviceTest().sources()
        args = spec.arguments
        short = lambda r, **kw: SimpleNamespace(**{**vars(args(r, **kw)), "horizon": 100})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", return_value={"workers": 1, "evaluation_episodes": 1}), \
                    patch.object(spec, "arguments", side_effect=short), patch.object(experiment.crossed, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                    patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
                    patch.object(experiment.source.learning, "update_mean", side_effect=AssertionError("unexpected update")), \
                    patch.object(experiment.source.learning, "final_checkpoint", side_effect=AssertionError("unexpected checkpoint")):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=False))
                self.assertEqual(len(list(Path(directory).rglob("*.pt"))), 0)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["rows"][0]["responses"]["zero"]["command_delta_rms"] = .1
                with self.assertRaisesRegex(ValueError, "zero-action"):
                    experiment.qualify(bad, preflight=False)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["rows"][0]["probe_states"] -= 1
                with self.assertRaises(ValueError):
                    experiment.qualify(bad, preflight=False)

    def test_fresh_rosters_all_roots_and_dynamic_scheduler(self):
        seen = set()
        for pref in (True, False):
            for root in spec.roots(preflight=pref):
                seeds = set(spec.seed_roles(root, preflight=pref)["native_evaluation"])
                self.assertFalse(seen & seeds)
                self.assertFalse(seeds & set(spec.source.seed_roles(root, preflight=pref)["native_evaluation"]))
                seen |= seeds
        self.assertEqual(len(spec.PROBES), 25)
        self.assertEqual(spec.budget(preflight=False)["native_steps"]*8, 153600)
        self.assertEqual(spec.budget(preflight=False)["probe_lower_mean_rows"]*8, 768000)
        full = task_specification("run", 410011, preflight=False)
        self.assertEqual((full["cpu"], full["ram_mb"], full["require_node"]), (5, 4096, None))
        self.assertEqual(full["allowed_nodes"], [f"node{i:03}" for i in range(1,7)])
        self.assertEqual(len(qualification_task("run", preflight=False)["wait_for_files"]), 8)
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            experiment.aggregate([], preflight=False)


if __name__ == "__main__":
    unittest.main()
