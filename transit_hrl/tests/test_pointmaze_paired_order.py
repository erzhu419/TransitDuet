import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_paired_order as experiment
from scripts import pointmaze_paired_order_stage104_spec as spec
from scripts.submit_pointmaze_paired_order_stage104_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_conditioned import all_seeds
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class PairedOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        source = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(source) for p in spec.PERIODS}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        return clones, predictor(), spec.source_record(410011), calibrations

    def test_shared_actor_roles_schedule_exact_budget_and_dynamic_resources(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = all_seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & set(all_seeds(spec.source.seed_roles(root, preflight=preflight))))
                self.assertFalse(set(seeds) & set(spec.comparison.seed_roles(root, preflight=preflight)["native_evaluation"]))
                seen.update(seeds)
                task = task_specification("unit_stage104", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage104", preflight=preflight)["result_dir"])
            n = spec.options(preflight=preflight)["updates"]
            for p in spec.PERIODS:
                schedule = spec.update_schedule(p, preflight=preflight)
                for method in spec.METHODS:
                    steps = [s for s in schedule if s["method"] == method]
                    for actor in ("upper", "lower"):
                        self.assertEqual([s["credit_iteration"] for s in steps if actor in s["allocation"]], list(range(1, n+1)))
                    phases = [s["phase"] for s in steps]
                    self.assertEqual(phases, ["joint"]*n if method in spec.JOINT_METHODS else ["lower"]*n+["upper"]*n)
                b = spec.matched_path_budget(p, preflight=preflight)
                self.assertEqual(b["joint"], b["staged"])
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (68608, 82329600))
        self.assertEqual((8*b["actor_mean_parameter_updates"], 8*b["checkpoint_writes"]), (1024, 64))
        self.assertEqual((b["policy_updates"], b["phase_boundary_checks"], b["upper_replay_forward_calls"]), (96, 8, 18432))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (304, 91200))
        self.assertEqual(len(spec.ENDPOINTS), 18)

    def test_real_updates_pair_first_lower_paths_and_freeze_both_staged_phases(self):
        source = self.sources()
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in source[0].items()}
        observed, counts, first_lower, lower_final = [], {}, {}, {}
        real_update = experiment.learning.update_mean

        def inspect_update(model, batches, **kw):
            self.assertIsNone(batches)
            credit = kw["actor_batches"]
            self.assertEqual(set(credit), set(kw["allocation"]))
            key = (kw["period"], kw["method"])
            counts[key] = counts.get(key, 0)+1
            for actor, batches in credit.items():
                conditioned = actor == "lower" and kw["method"] in ("joint_conditioned", "staged_common")
                for groups in batches.values():
                    for group in groups:
                        self.assertTrue(all(("upper_noise_seed" in row) == conditioned for _, row in group))
            if counts[key] == 1:
                first_lower[key] = [(b.lower.state.copy(), r["episode_return"], r["policy_seed"], r["lower_seed"])
                    for groups in credit["lower"].values() for group in groups for b, r in group]
                if kw["method"] in spec.STAGED_METHODS:
                    donor = "joint_conditioned" if kw["method"] == "staged_common" else "joint_independent"
                    for actual, expected in zip(first_lower[key], first_lower[(kw["period"], donor)]):
                        np.testing.assert_array_equal(actual[0], expected[0])
                        self.assertEqual(actual[1:], expected[1:])
            before = copy.deepcopy(model.state_dict())
            if kw["method"] in spec.STAGED_METHODS and set(credit) == {"upper"}:
                torch.testing.assert_close(before["lower_actor"], lower_final[key], atol=0, rtol=0)
            result = real_update(model, None, **kw)
            experiment.learning.check_training_freeze(model, before, kw["allocation"])
            if kw["method"] in spec.STAGED_METHODS and counts[key] == 2:
                lower_final[key] = copy.deepcopy(model.state_dict()["lower_actor"])
            for row in result["actors"].values():
                self.assertEqual(row["gradient_episodes"], 8)
                self.assertLess(row["max_abs_old_logp_difference"], 1e-4)
            observed.append(key)
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment.joint.fresh, "load_source", return_value=source), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=inspect_update), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=True, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(len(observed), 24)
                self.assertEqual(experiment.aggregate([cell], preflight=True)["paired_order_confirmation"], "mechanical_only")
                for g in cell["groups"].values():
                    self.assertTrue(all(t["checkpoint"] is None for t in g["trained"].values()))
                    self.assertTrue(all("upper_noise_seed" not in r and r["upper_replay_forward_calls"] == 0
                        for rows in g["evaluation"].values() for r in rows))
                for field, value in (("phase", "upper"), ("credit_iteration", 2), ("phase_boundary_freeze", "passed")):
                    bad = copy.deepcopy(cell)
                    bad["groups"]["50"]["trained"]["staged_common"]["history"][0][field] = value
                    with self.assertRaisesRegex(ValueError, "phase or shared credit roster"):
                        experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["cost"]["phase_boundary_checks"] -= 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["joint_conditioned"][0]["upper_noise_seed"] = 1
                with self.assertRaisesRegex(ValueError, "leaked into evaluation"):
                    experiment.qualify(bad, preflight=True)
        for p, model in source[0].items(): experiment.learning.native.curves.support.assert_frozen(model, snapshots[p])

    def test_reduced_full_fixture_writes_only_final_policies_after_both_phases(self):
        real_options, real_args = spec.options, spec.arguments
        small = lambda **kw: {**real_options(**kw), "updates": 1, "credit_scenarios_per_batch": 2, "evaluation_episodes": 1, "workers": 1}
        args = lambda root, **kw: SimpleNamespace(**{**vars(real_args(root, **kw)), "horizon": 200})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=args), \
                    patch.object(experiment.joint.fresh, "load_source", return_value=self.sources()), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"]["checkpoint_writes"], 8)
                self.assertEqual(len(list((Path(directory)/"final_weights").glob("*.pt"))), 8)
                for p, g in cell["groups"].items():
                    for method, trained in g["trained"].items():
                        payload = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
                        self.assertEqual((payload["protocol"], payload["period"], payload["method"], payload["updates"]),
                            (spec.EXPERIMENT_PROTOCOL, int(p), method, 1))
                        self.assertEqual(len(trained["history"]), 1 if method in spec.JOINT_METHODS else 2)
                self.assertTrue((Path(directory)/"completion/ready.json").exists())

    def test_primary_gate_keeps_both_periods_and_all18_comparisons(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/36, 1-.05/36])
            self.assertEqual(result["paired_order_confirmation"], "supported")
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 0.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["paired_order_confirmation"], "not_supported")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
