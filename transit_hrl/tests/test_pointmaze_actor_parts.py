import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_actor_parts as experiment
from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts import pointmaze_actor_parts_stage81_spec as spec
from scripts.submit_pointmaze_actor_parts_stage81_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class ActorPartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_masks_partition_mean_and_scale_without_changing_gradient(self):
        sigma = np.array([True, True, False, False, False])
        g = np.array([1., -2., 3., -4., 5.])
        parts = experiment.masks(sigma)
        np.testing.assert_array_equal(np.where(parts["mean"], g, 0.) + np.where(parts["log_std"], g, 0.), g)
        self.assertFalse(np.any(parts["mean"] & parts["log_std"]))
        self.assertTrue(parts["full"].all())

    def test_fixed_radius_candidates_change_only_registered_part_and_keep_source(self):
        actor = GaussianActor(1, 1, 0, -.5)
        before = copy.deepcopy(actor.state_dict())
        states = np.linspace(.1, 1., 12, dtype=np.float32).reshape(-1, 1)
        g, sigma = np.array([1., -2., -3.]), np.array([True, False, False])
        for part, mask in experiment.masks(sigma).items():
            changed, geometry, cost = experiment.native.direction.matched_perturbations(actor, states,
                np.where(mask, g, 0.), delta=spec.FISHER_RADIUS, chunk_size=5)
            self.assertEqual(geometry["radius_check"], "passed")
            self.assertEqual(cost, {"fisher_jvp_batches": 3, "exact_kl_forward_batches": 6})
            for candidate in changed.values():experiment.check_parameter_part(actor, candidate, part)
            std = experiment.std_summary(actor, changed)
            if part == "mean":
                for row in std["candidates"].values():self.assertEqual(row["std_ratio"], [1.])
            if part == "log_std":
                self.assertLess(std["candidates"]["plus"]["std_ratio"][0], 1.)
                self.assertGreater(std["candidates"]["minus"]["std_ratio"][0], 1.)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        bad = copy.deepcopy(actor)
        with torch.no_grad():bad.log_std.add_(.01)
        with self.assertRaises(AssertionError):experiment.check_parameter_part(actor, bad, "mean")
        bad = copy.deepcopy(actor)
        with torch.no_grad():bad.net[0].bias.add_(.01)
        with self.assertRaises(AssertionError):experiment.check_parameter_part(actor, bad, "log_std")

    def test_shared_credit_generates_all_parts_with_no_source_or_other_actor_update(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(model.state_dict())
        args, roles = spec.arguments(310011, preflight=True), spec.seed_roles(310011, preflight=True)
        collector = scenario_fixture.ScenarioCreditTest()
        batches = {name: collector.collect(model, args, roles["credit_" + name]) for name in ("A", "B")}
        cost = dict.fromkeys(spec.budget(preflight=True), 0)
        candidates, credit = experiment.credit_directions(model, batches, period=50, horizon=args.horizon, cost=cost)
        experiment.native.curves.support.assert_frozen(model, before)
        self.assertEqual(set(candidates), set(spec.VARIANTS) - {"base", "zero"})
        self.assertEqual(cost["objective_checks"], 8)
        self.assertEqual(cost["mc_calls"], 16)
        self.assertEqual(cost["actor_score_forward_batches"], 16)
        self.assertEqual(cost["actor_score_backward_batches"], 48)
        self.assertEqual(cost["fisher_jvp_batches"], 12)
        self.assertEqual(cost["exact_kl_forward_batches"], 24)
        self.assertEqual(cost["parameter_part_checks"], 12)
        for a in ("upper", "lower"):
            untouched = "lower_actor" if a == "upper" else "upper_actor"
            for part in spec.PARTS:
                for sign in ("plus", "minus"):
                    w = candidates[f"{a}_{part}_{sign}"]
                    torch.testing.assert_close(w[untouched], before[untouched], atol=0, rtol=0)
                    if part == "mean":torch.testing.assert_close(w[a+"_actor"]["log_std"], before[a+"_actor"]["log_std"], atol=0, rtol=0)
                    if part == "log_std":
                        for k, v in before[a+"_actor"].items():
                            if k != "log_std":torch.testing.assert_close(w[a+"_actor"][k], v, atol=0, rtol=0)
                row = credit["actors"][a]["parts"][part]
                self.assertEqual(row["parameter_part_check"], "passed")
                for noise in row["scenario_group_noise"].values():self.assertEqual(noise["episodes"], 2)

    def test_preregistered_fresh_roster_fixed_budget_and_dynamic_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"] * 8, b["native_steps"] * 8), (8192, 9830400))
        self.assertEqual(b["actor_score_backward_batches"] * 8, 9216)
        self.assertEqual(b["fisher_jvp_batches"] * 8, 3672)
        self.assertEqual(b["parameter_part_checks"] * 8, 192)
        self.assertEqual(len(spec.ENDPOINTS), 62)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                all_roles = [s["scenario_seed"] for name in ("A", "B") for s in roles["credit_" + name]]
                all_roles += [n for name in ("A", "B") for s in roles["credit_" + name] for n in s["noise_seeds"]]
                all_roles += roles["native_evaluation"]
                self.assertEqual(len(all_roles), len(set(all_roles)))
                self.assertFalse(set(all_roles) & set(spec.source.seed_roles(root, preflight=preflight)["native_evaluation"]))
                task = task_specification("unit_stage81", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage81", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])

    def test_all62_endpoints_share_frozen_root_bootstrap_family_and_hold(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda cell, **kw: cell), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.native.np, "quantile", wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(quantile.call_args.args[1], [.05 / 124, 1 - .05 / 124])
        self.assertEqual(set(summary["endpoints"]), set(spec.ENDPOINTS))
        self.assertEqual(summary["cost"]["native_episodes"], 8192)
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
