import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeDriver
from freq_hrl.experiments import pointmaze_conditioning_shift as experiment
from scripts import pointmaze_conditioning_shift_stage105_spec as spec
from scripts import pointmaze_paired_order_stage104_spec as previous
from scripts.submit_pointmaze_conditioning_shift_stage105_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_conditioned import all_seeds
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class ConditioningShiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        model = fixture.FeasibleCreditTest().source()
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        return ({str(p): copy.deepcopy(model) for p in spec.PERIODS}, predictor(), spec.source_record(410011),
            {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS})

    def test_one_task_variable_fresh_roles_budget_and_dynamic_resources(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                original = vars(spec.source.arguments(root, preflight=preflight))
                changed = vars(spec.arguments(root, preflight=preflight))
                self.assertEqual(changed["regime_dwell_seconds"], (.4, .8))
                self.assertEqual({k: v for k, v in original.items() if k != "regime_dwell_seconds"},
                    {k: v for k, v in changed.items() if k != "regime_dwell_seconds"})
                seeds = all_seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & set(all_seeds(previous.seed_roles(root, preflight=preflight))))
                seen.update(seeds)
                t = task_specification("unit_stage105", root, preflight=preflight)
                self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(t.get("require_node"))
                self.assertEqual((t["cpu"], t["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertIn(spec.RUNNER_SCRIPT, t["cmd"])
                self.assertTrue(t["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage105", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (34816, 41779200))
        self.assertEqual((8*b["actor_mean_parameter_updates"], 8*b["checkpoint_writes"]), (512, 32))
        self.assertEqual((b["credit_episodes"], b["evaluation_episodes"], b["upper_replay_forward_calls"]), (4096, 256, 9216))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["upper_replay_forward_calls"]), (160, 72))

    def test_driver_changes_regimes_without_changing_nuisances(self):
        baseline = PointMazeRegimeDriver(seed=105001, horizon=1200, dt_seconds=.01)
        shifted = PointMazeRegimeDriver(seed=105001, horizon=1200, dt_seconds=.01,
            regime_dwell_seconds=spec.REGIME_DWELL_SECONDS)
        intervals = np.diff((0, *shifted._regime_change_steps))
        self.assertTrue(np.all((intervals >= 40) & (intervals <= 80)))
        self.assertGreater(len(shifted._regime_change_steps), len(baseline._regime_change_steps))
        np.testing.assert_array_equal(shifted._forces, baseline._forces)
        np.testing.assert_array_equal(shifted._distractors, baseline._distractors)
        self.assertEqual(shifted.start_vertex, baseline.start_vertex)
        self.assertFalse(np.array_equal(shifted._targets, baseline._targets))

    def test_shift_reaches_every_rollout_and_evaluation_stays_independent(self):
        sources = self.sources()
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in sources[0].items()}
        observed = []
        real_update = experiment.learning.update_mean

        def make_task(**kw):
            expected = spec.task_options(410011, preflight=True)
            for key, value in expected.items():
                self.assertEqual(list(kw[key]) if isinstance(kw[key], tuple) else kw[key], value)
            return DenseTask()

        def inspect_update(model, batches, **kw):
            credit = kw["actor_batches"]
            for actor, pools in credit.items():
                for groups in pools.values():
                    for group in groups:
                        for _, r in group:
                            self.assertEqual("upper_noise_seed" in r, actor == "lower" and kw["method"] == "joint_conditioned")
            observed.append(kw["method"])
            return real_update(model, batches, **kw)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment.joint.fresh, "load_source", return_value=sources), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=inspect_update), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=make_task) as tasks, \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=True, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(tasks.call_count, cell["cost"]["native_episodes"])
                self.assertEqual(len(observed), 8)
                self.assertEqual(experiment.aggregate([cell], preflight=True)["shifted_conditioning_confirmation"], "mechanical_only")
                for g in cell["groups"].values():
                    self.assertTrue(all(t["checkpoint"] is None for t in g["trained"].values()))
                    self.assertTrue(all("upper_noise_seed" not in r and r["upper_replay_forward_calls"] == 0
                        for rows in g["evaluation"].values() for r in rows))
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["task_options"]["regime_dwell_seconds"] = [.8, 1.6]
                with self.assertRaisesRegex(ValueError, "Shifted task"):
                    experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["100"]["evaluation"]["joint_conditioned"][0]["upper_noise_seed"] = 1
                with self.assertRaisesRegex(ValueError, "leaked"):
                    experiment.qualify(bad, preflight=True)
        for p, model in sources[0].items():
            experiment.learning.native.curves.support.assert_frozen(model, snapshots[p])

    def test_reduced_full_writes_four_fixed_final_models(self):
        options, args = spec.options, spec.arguments
        small = lambda **kw: {**options(**kw), "updates": 1, "credit_scenarios_per_batch": 2, "evaluation_episodes": 1, "workers": 1}
        short = lambda root, **kw: SimpleNamespace(**{**vars(args(root, **kw)), "horizon": 200})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=short), \
                    patch.object(experiment.joint.fresh, "load_source", return_value=self.sources()), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"]["checkpoint_writes"], 4)
                self.assertEqual(len(list((Path(directory)/"final_weights").glob("*.pt"))), 4)
                for p, g in cell["groups"].items():
                    for m, t in g["trained"].items():
                        saved = torch.load(t["checkpoint"], map_location="cpu", weights_only=False)
                        self.assertEqual((saved["protocol"], saved["period"], saved["method"], saved["updates"]),
                            (spec.EXPERIMENT_PROTOCOL, int(p), m, 1))

    def test_all_four_primaries_and_all12_contrasts_are_required(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(len(result["endpoints"]), 12)
            self.assertEqual(quantile.call_args.args[1], [.05/24, 1-.05/24])
            self.assertEqual(result["shifted_conditioning_confirmation"], "supported")
            for k in spec.PRIMARY_ENDPOINTS:
                for c in cells: c["groups"]["both"]["effects"][k] = 0.
                self.assertEqual(experiment.aggregate(cells, preflight=False)["shifted_conditioning_confirmation"], "not_supported")
                for c in cells: c["groups"]["both"]["effects"][k] = 2.
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
