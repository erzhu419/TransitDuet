import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_plan_baselines as experiment
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch
from scripts import pointmaze_plan_baselines_stage106_spec as spec
from scripts.submit_pointmaze_plan_baselines_stage106_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_conditioned import all_seeds
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class PlanBaselinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        model = fixture.FeasibleCreditTest().source()
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        return ({str(p): copy.deepcopy(model) for p in spec.PERIODS}, predictor(), spec.source_record(410011),
            {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS})

    def test_registered_rosters_full_sample_budget_and_dynamic_placement(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = all_seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & set(all_seeds(spec.source.seed_roles(root, preflight=preflight))))
                seen.update(seeds)
                t = task_specification("unit_stage106", root, preflight=preflight)
                self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(t.get("require_node"))
                self.assertEqual((t["cpu"], t["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertIn(spec.RUNNER_SCRIPT, t["cmd"])
                self.assertTrue(t["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage106", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (69120, 82944000))
        self.assertEqual((8*b["actor_mean_parameter_updates"], 8*b["checkpoint_writes"]), (768, 64))
        self.assertEqual((b["credit_episodes"], b["evaluation_episodes"], b["upper_replay_forward_calls"]), (8192, 448, 9216))
        self.assertEqual(spec.budget(preflight=True)["native_episodes"], 312)
        for p in spec.PERIODS:
            for m in spec.METHODS:
                self.assertAlmostEqual(sum(v/(p if a == "upper" else 1) for a, v in spec.allocation(m, p).items()), 1.)

    def test_flat_full_feedback_causal_no_upper_or_forecast_and_primitive_credit(self):
        sources = self.sources()
        model = sources[0]["50"]
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        weights = experiment.native.joint.inference_weights(model)
        with patch.object(experiment.native, "_WORKER", (model, args)), \
                patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
                patch.object(model, "act_upper", side_effect=AssertionError("flat invoked upper")), \
                patch.object(experiment.forecast, "plan_points", side_effect=AssertionError("flat invoked forecast")):
            pairs = []
            for shift in (0., 1.):
                task = DenseTask(shift)
                native_task = SimpleNamespace(environment=task.environment, reset=task.reset, step=task.step,
                    action_low=task.action_low, action_high=task.action_high)
                with patch.object(experiment.native.joint, "_make_task", return_value=native_task):
                    pairs.append(experiment.primitive_episode((weights, 106001, 106002, "flat_lower", 50, None, .02, {}, True)))
            b, row = pairs[0]
            self.assertIsInstance(b, LevelTrajectoryBatch)
            self.assertFalse(hasattr(b, "upper"))
            experiment.check_row(row, 50, 100)
            np.testing.assert_array_equal(b.state[:50], pairs[1][0].state[:50])
            self.assertFalse(np.array_equal(b.state[51:], pairs[1][0].state[51:]))
            np.testing.assert_array_equal(b.state[0, -2:], [0., 0.])
            np.testing.assert_array_equal(np.flatnonzero(b.done), [99])
            r = experiment.native.independent.exact_returns(b, 1.)
            self.assertAlmostEqual(r[0], row["episode_return"], places=4)

    def test_forecast_matches_zero_execution_without_upper_inference(self):
        sources = self.sources()
        model, pred, _, cal = sources[0]["50"], sources[1], sources[2], sources[3]["50"]
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        weights = experiment.native.joint.inference_weights(model)
        policy_seed, lower_seed = experiment.scenario.spec.noise_seeds(410011, 106001, 106002)
        with patch.object(experiment.native, "_WORKER", (model, args)), \
                patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            zero, zr = experiment.native.native_episode(weights, seed=106001, variant="zero", period=50,
                predictor=pred, alpha=0., envelope=cal["envelope"], collect=True, policy_seed=policy_seed, lower_seed=lower_seed)
            with patch.object(model, "act_upper", side_effect=AssertionError("forecast invoked upper")):
                batch, row = experiment.primitive_episode((weights, 106001, 106002, "forecast_lower", 50, pred, .02, cal["envelope"], True))
            for k in ("state", "action", "reward", "old_logp", "old_value", "value_state"):
                np.testing.assert_array_equal(getattr(batch, k), getattr(zero.lower, k))
            self.assertEqual(row["episode_return"], zr["episode_return"])
            self.assertEqual(row["upper_calls"], 0)
            self.assertEqual(row["plan_renewals"], 2)

    def test_reduced_full_all_learners_real_scores_frozen_sources_and_final_only(self):
        options, args = spec.options, spec.arguments
        small = lambda **kw: {**options(**kw), "updates": 1, "credit_scenarios_per_batch": 2, "evaluation_episodes": 1, "workers": 1}
        short = lambda root, **kw: SimpleNamespace(**{**vars(args(root, **kw)), "horizon": 200})
        sources, updates = self.sources(), []
        before = {p: copy.deepcopy(m.state_dict()) for p, m in sources[0].items()}
        real = experiment.learning.update_mean

        def inspect(model, batches, **kw):
            updates.append(kw["method"])
            if kw["method"].startswith("joint_"):
                for actor, pools in kw["actor_batches"].items():
                    for groups in pools.values():
                        for group in groups:
                            for _, r in group:
                                self.assertEqual("upper_noise_seed" in r, actor == "lower" and kw["method"] == "joint_conditioned")
            else:
                self.assertEqual(kw["allocation"], {"lower": 1.})
                for groups in batches.values():
                    self.assertEqual(len(groups), 4)
                    self.assertTrue(all(isinstance(b, LevelTrajectoryBatch) for group in groups for b, _ in group))
            return real(model, batches, **kw)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=short), \
                    patch.object(experiment.joint.fresh, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=inspect), \
                    patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()) as tasks, \
                    patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=False))
                self.assertEqual(cell["native_planning_cost"], spec.planning_budget(preflight=False))
                self.assertEqual(tasks.call_count, cell["cost"]["native_episodes"])
                self.assertEqual(len(updates), 8)
                self.assertEqual(len(list((Path(directory)/"final_weights").glob("*.pt"))), 8)
                for p, g in cell["groups"].items():
                    for m, t in g["trained"].items():
                        saved = torch.load(t["checkpoint"], map_location="cpu", weights_only=False)
                        self.assertEqual((saved["protocol"], saved["period"], saved["method"], saved["updates"]),
                            (spec.EXPERIMENT_PROTOCOL, int(p), m, 1))
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["flat_lower"][0]["plan_ols_fits"] = 1
                with self.assertRaisesRegex(ValueError, "planning calls"):
                    experiment.qualify(bad, preflight=False)
                bad = copy.deepcopy(cell)
                bad["groups"]["100"]["evaluation"]["joint_conditioned"][0]["upper_noise_seed"] = 1
                with self.assertRaisesRegex(ValueError, "leaked"):
                    experiment.qualify(bad, preflight=False)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["trained"]["flat_lower"]["history"][0]["actors"]["lower"]["gradient_episodes"] //= 2
                with self.assertRaisesRegex(ValueError, "MC credit"):
                    experiment.qualify(bad, preflight=False)
        for p, model in sources[0].items():
            experiment.native.curves.support.assert_frozen(model, before[p])

    def test_all26_contrasts_and_four_primaries_required(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(len(result["endpoints"]), 26)
            self.assertEqual(quantile.call_args.args[1], [.05/52, 1-.05/52])
            self.assertEqual(result["plan_baseline_confirmation"], "supported")
            for k in spec.PRIMARY_ENDPOINTS:
                for c in cells:
                    c["groups"]["both"]["effects"][k] = 0.
                self.assertEqual(experiment.aggregate(cells, preflight=False)["plan_baseline_confirmation"], "not_supported")
                for c in cells:
                    c["groups"]["both"]["effects"][k] = 2.
            with self.assertRaises(ValueError):
                experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
