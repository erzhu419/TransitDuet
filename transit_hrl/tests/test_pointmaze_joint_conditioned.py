import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_conditioned as experiment
from scripts import pointmaze_joint_conditioned_stage102_spec as spec
from scripts.submit_pointmaze_joint_conditioned_stage102_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


def all_seeds(roles):
    values = [v for rows in roles["training_rounds"] for roster in rows.values() for row in roster
        for v in (row["scenario_seed"], *row["noise_seeds"])]
    return values + roles["native_evaluation"]


class JointConditionedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actor_roles_budget_and_dynamic_resources(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                current = all_seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(current), len(set(current)))
                self.assertFalse(set(current) & seen)
                seen.update(current)
                self.assertFalse(set(current) & set(all_seeds(spec.source.seed_roles(root, preflight=preflight))))
                task = task_specification("unit_stage102", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage102", preflight=preflight)["result_dir"])
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (35840, 43008000))
        self.assertEqual((8*b["actor_mean_parameter_updates"], 8*b["checkpoint_writes"]), (512, 32))
        self.assertEqual(b["upper_independent_pair_checks"], b["lower_independent_pair_checks"]+b["lower_common_pair_checks"])
        self.assertEqual(b["upper_replay_forward_calls"], 9216)
        self.assertEqual(len(spec.ENDPOINTS), 20)
        for p in spec.PERIODS:
            self.assertEqual(spec.allocation("joint_independent", p), spec.allocation("joint_conditioned", p))
            a = spec.allocation("joint_conditioned", p)
            self.assertAlmostEqual(spec.FISHER_RADIUS*(a["lower"]+a["upper"]/p), .001, places=12)

    def test_real_joint_updates_keep_credit_pools_separate_and_source_frozen(self):
        root = 410011
        source = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(source) for p in spec.PERIODS}
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in clones.items()}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        observations = []
        real_update = experiment.learning.update_mean

        def inspect_update(model, batches, **kw):
            self.assertIsNone(batches)
            credit = kw["actor_batches"]
            self.assertEqual(set(credit), {"upper", "lower"})
            upper_seeds, lower_seeds = set(), set()
            for actor in credit:
                for groups in credit[actor].values():
                    for group in groups:
                        for _, row in group:
                            (upper_seeds if actor == "upper" else lower_seeds).add(row["seed"])
                            conditioned = actor == "lower" and kw["method"] == "joint_conditioned"
                            self.assertEqual("upper_noise_seed" in row, conditioned)
                            if not conditioned: self.assertEqual(row["upper_replay_forward_calls"], 0)
            self.assertFalse(upper_seeds & lower_seeds)
            before = copy.deepcopy(model.state_dict())
            with patch.object(experiment.learning.parts, "scenario_actor_scores",
                    wraps=experiment.learning.parts.scenario_actor_scores) as scoring:
                result = real_update(model, batches, **kw)
                self.assertEqual(scoring.call_count, 2)
                for call in scoring.call_args_list:
                    actor = call.kwargs["actor_names"][0]
                    self.assertIs(call.args[1], credit[actor])
            with self.assertRaisesRegex(ValueError, "exactly the updated actors"):
                real_update(model, None, **{**kw, "actor_batches": {"upper": credit["upper"]}})
            experiment.learning.check_training_freeze(model, before, ("upper", "lower"))
            for actor in ("upper", "lower"):
                self.assertTrue(any(not torch.equal(v, model.state_dict()[actor+"_actor"][k])
                    for k, v in before[actor+"_actor"].items()))
                self.assertEqual(result["actors"][actor]["gradient_episodes"], 8)
                self.assertLess(result["actors"][actor]["max_abs_old_logp_difference"], 1e-4)
            observations.append(kw["method"])
            bad = copy.deepcopy(credit["upper"]["A"][0])
            bad[0][1]["upper_noise_seed"] = bad[0][1]["noise_seed"]
            roster = {"scenario_seed": bad[0][1]["seed"], "noise_seeds": [r["noise_seed"] for _, r in bad]}
            with self.assertRaisesRegex(ValueError, "replayed upper noise"):
                experiment.check_independent_pair(bad, roster, root=root)
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment.fresh, "load_source", return_value=(clones, predictor(), spec.source_record(root), calibrations)), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=inspect_update), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(root, preflight=True, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(len(observations), 8)
                self.assertEqual(experiment.aggregate([cell], preflight=True)["joint_conditioning_confirmation"], "mechanical_only")
                for p, g in cell["groups"].items():
                    for rows in g["evaluation"].values():
                        self.assertTrue(all("upper_noise_seed" not in r and r["upper_replay_forward_calls"] == 0 for r in rows))
                    self.assertTrue(all(t["checkpoint"] is None for t in g["trained"].values()))
                bad = copy.deepcopy(cell)
                bad["cost"]["upper_independent_pair_checks"] -= 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["training_credit"] = "shared_upper_baseline"
                with self.assertRaisesRegex(ValueError, "credit contract"): experiment.qualify(bad, preflight=True)
        for p, model in clones.items():
            experiment.learning.native.curves.support.assert_frozen(model, snapshots[p])

    def test_primary_gate_requires_control_and_original_teacher_at_both_periods(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/40, 1-.05/40])
            self.assertEqual(result["joint_conditioning_confirmation"], "supported")
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 0.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["joint_conditioning_confirmation"], "not_supported")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
