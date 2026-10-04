import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_action_gain as experiment
from scripts import pointmaze_action_gain_stage110_spec as spec
from scripts.submit_pointmaze_action_gain_stage110_scheduleurm import task_specification, qualification_task
import test_pointmaze_crossed_advice as fixture
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


class ActionGainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_direction_changes_only_one_coordinate_and_not_input(self):
        action = np.array([.1, .2, -.3, .4], dtype=np.float32)
        before = action.copy()
        for i in range(4):
            for sign in ("plus", "minus"):
                expected = before.copy()
                expected[i] += spec.EPSILON if sign == "plus" else -spec.EPSILON
                np.testing.assert_array_equal(experiment.offset_action(action, f"axis{i}_{sign}"), expected)
        np.testing.assert_array_equal(action, before)

    def test_known_quadratic_return_gives_exact_slope_and_curvature(self):
        returns = dict.fromkeys(spec.VARIANTS, 10.)
        returns["axis0_plus"] += 2*spec.EPSILON-3*spec.EPSILON**2
        returns["axis0_minus"] += -2*spec.EPSILON-3*spec.EPSILON**2
        pair = {"evaluation": {v: {"episode_return": r} for v, r in returns.items()}}
        e = experiment.effects(50, [pair, pair])
        self.assertEqual(len(e), 18)
        self.assertEqual(e["50/axis0_slope"], 2.)
        self.assertEqual(e["50/axis0_curvature"], -6.)
        self.assertEqual(e["50/axis0_plus_minus_forecast"], .3125)

    def test_native_pair_has_exact_zero_identity_and_preserves_sources(self):
        sources = fixture.CrossedAdviceTest().sources()
        model, pred = sources[0]["50"]["learned_hint"], sources[1]
        before = copy.deepcopy(model.state_dict())
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        experiment.source.native.init_worker(model.config, args)
        with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            pair = experiment.worker_pair((experiment.source.native.joint.inference_weights(model), 110095001, 50, pred, sources[3]["50"]["envelope"]))
        self.assertEqual(pair["pair_checks"], "passed")
        self.assertLessEqual(pair["lower_innovation_max_error"], 3e-5)
        self.assertEqual(pair["evaluation"]["zero"]["episode_return"], pair["evaluation"]["forecast"]["episode_return"])
        for v, r in pair["evaluation"].items():
            experiment.check_row(r, root=410011, period=50, horizon=100, variant=v)
        self.assertNotEqual(pair["evaluation"]["axis0_plus"]["episode_return"], pair["evaluation"]["axis0_minus"]["episode_return"])
        experiment.source.native.curves.support.assert_frozen(model, before)

    def test_reduced_run_exact_costs_no_training_or_trace_artifacts(self):
        sources = fixture.CrossedAdviceTest().sources()
        args = spec.arguments
        short = lambda r, **kw: SimpleNamespace(**{**vars(args(r, **kw)), "horizon": 100})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", return_value={"workers": 1, "paths_per_panel": 1}), \
                    patch.object(spec, "arguments", side_effect=short), patch.object(experiment.crossed, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                    patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
                    patch.object(experiment.source.learning, "update_mean", side_effect=AssertionError("unexpected training")), \
                    patch.object(experiment.source.learning, "final_checkpoint", side_effect=AssertionError("unexpected checkpoint")):
                cell = experiment.run(410011, preflight=True, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                summary = experiment.aggregate([cell], preflight=True)
                self.assertEqual(len(summary["endpoints"]), 36)
                self.assertEqual(summary["closed_loop_gain"], "mechanical_only")
                self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 0)
                self.assertEqual(len(list(Path(directory).rglob("*.pt"))), 0)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["panels"]["B"]["pairs"][0]["seed"] += 1
                with self.assertRaisesRegex(ValueError, "panel"):
                    experiment.qualify(bad, preflight=True)

    def test_gain_gate_needs_both_panels_and_corrected_slope(self):
        endpoints = {k: {"mean": 0., "ci": [-1., 1.]} for k in spec.ENDPOINTS}
        for p in spec.PERIODS:
            endpoints[f"{p}/axis0_plus_minus_forecast"] = {"mean": 1., "ci": [.5, 1.5]}
            endpoints[f"{p}/axis0_slope"] = {"mean": 2., "ci": [1., 3.]}
        cells = [{"groups": {str(p): {"effects": {f"{p}/{k}": 1. for k in spec.METRICS},
            "panels": {panel: {"effects": {f"{p}/{k}": 1. for k in spec.METRICS}} for panel in spec.PANELS}}
            for p in spec.PERIODS}} for _ in range(8)]
        with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
            self.assertEqual(experiment.aggregate(cells, preflight=False)["closed_loop_gain"], "detected_both_periods")
            for c in cells:
                c["groups"]["50"]["panels"]["B"]["effects"]["50/axis0_slope"] = -1.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["closed_loop_gain"], "partial")
            endpoints["100/axis0_slope"]["ci"] = [-1., 3.]
            self.assertEqual(experiment.aggregate(cells, preflight=False)["closed_loop_gain"], "not_supported")

    def test_fresh_panels_and_dynamic_budget(self):
        seen = set()
        for pref in (True, False):
            for root in spec.roots(preflight=pref):
                for seeds in spec.seed_roles(root, preflight=pref)["panels"].values():
                    self.assertFalse(seen.intersection(seeds))
                    self.assertFalse(set(seeds).intersection(spec.source.seed_roles(root, preflight=pref)["native_evaluation"]))
                    seen.update(seeds)
        self.assertEqual(spec.budget(preflight=False)["native_episodes"]*8, 3072)
        self.assertEqual(spec.budget(preflight=False)["native_steps"]*8, 3686400)
        self.assertEqual(spec.budget(preflight=False)["native_upper_calls"]*8, 41472)
        task = task_specification("run", 410011, preflight=False)
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (5, 4096, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1,7)])
        self.assertEqual(len(qualification_task("run", preflight=False)["wait_for_files"]), 8)
        with self.assertRaises(ValueError):
            experiment.aggregate([], preflight=False)


if __name__ == "__main__":
    unittest.main()
