import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import pointmaze_crossed_direction as experiment
from scripts import pointmaze_crossed_direction_stage74_spec as spec
from scripts.submit_pointmaze_crossed_direction_stage74_scheduleurm import task_specification, qualification_task


def evaluation(seeds):
    result = {}
    for ei, e in enumerate(spec.EXECUTION_ARMS):
        rows = {v: [{"seed": seed, "episode_return": float(i), "policy_seed": seed + 1, "lower_seed": seed + 2,
            "decision_steps": [0, 50], "upper_proposed_actions": [[0., 0., 0., 0.]] * 2} for i, seed in enumerate(seeds)]
            for v in spec.VARIANTS}
        for fi, f in enumerate(spec.FIT_ARMS):
            for di, d in enumerate(spec.DIRECTIONS):
                gain = (1 + fi + ei * (3 + 4 * fi)) * (di + 1)
                for sign, scale in (("plus", 1.), ("minus", -.5)):
                    for row in rows[spec.variant(f, d, sign)]:row["episode_return"] += scale * gain
        result[e] = rows
    return result


def fixture(root, *, preflight):
    roles = spec.seed_roles(root, preflight=preflight)
    groups = {}
    for p in spec.PERIODS:
        ev = evaluation(roles["native_evaluation"])
        groups[str(p)] = {"fitting": {f: {"geometry": {d: {} for d in spec.DIRECTIONS},
            "model_and_Adam_unchanged": "passed", "stage73_direction_reproduction": "passed"} for f in spec.FIT_ARMS},
            "evaluation": ev, "effects": experiment.paired_endpoints(str(p), ev, roles["native_evaluation"]),
            "pairing": "passed", "source_and_Adam_unchanged": "passed", "candidate_reuse_across_execution": "same_parameters_and_step"}
    return {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": spec.budget(preflight=preflight), "groups": groups,
        "native_planning_cost": {"plan_ols_fits": 2}, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0}


class CrossedDirectionTest(unittest.TestCase):
    def test_cells_execution_fitting_and_interaction_use_correct_signs(self):
        effects = experiment.paired_endpoints("50", evaluation([11, 12, 13]), [11, 12, 13])
        d = "gae_control"
        self.assertEqual(effects[f"cell/50/zero_train/zero_train/{d}/plus_base"], 1.)
        self.assertEqual(effects[f"cell/50/joint_ppo/joint_ppo/{d}/plus_minus"], 13.5)
        self.assertEqual(effects[f"cell/50/joint_ppo/joint_ppo/{d}/minus_base"], -4.5)
        self.assertEqual(effects[f"execution/50/zero_train/{d}"], 3.)
        self.assertEqual(effects[f"execution/50/joint_ppo/{d}"], 7.)
        self.assertEqual(effects[f"fitting/50/zero_train/{d}"], 1.)
        self.assertEqual(effects[f"fitting/50/joint_ppo/{d}"], 5.)
        self.assertEqual(effects[f"interaction/50/{d}"], 4.)
        self.assertEqual(len(effects), 51)

    def test_cross_execution_pairing_rejects_noise_proposals_and_missing_variants(self):
        for key, value in (("lower_seed", 999), ("upper_proposed_actions", [[1., 0., 0., 0.]])):
            ev = evaluation([11, 12])
            ev["joint_ppo"]["base"][0][key] = value
            with self.assertRaises(ValueError):experiment.paired_endpoints("50", ev, [11, 12])
        ev = evaluation([11, 12])
        del ev["joint_ppo"]["base"]
        with self.assertRaises(ValueError):experiment.paired_endpoints("50", ev, [11, 12])
        ev = evaluation([11, 12])
        ev["zero_train"][spec.VARIANTS[1]].reverse()
        with self.assertRaises(ValueError):experiment.paired_endpoints("50", ev, [11, 12])

    def test_reproduction_requires_both_original_geometry_and_reward_frame(self):
        old = {"geometry": {"gae_control": {"step": .01}}, "native_baseline_frame": {"location": 1.}}
        experiment.check_reproduction(copy.deepcopy(old), old)
        for key in old:
            changed = copy.deepcopy(old)
            changed[key] = {}
            with self.assertRaises(ValueError):experiment.check_reproduction(changed, old)

    def test_qualification_detects_changed_cost_effect_and_optimization(self):
        original = fixture(310001, preflight=True)
        experiment.qualify(original, preflight=True)
        bad = copy.deepcopy(original)
        bad["cost"]["native_episodes"] -= 1
        with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(original)
        bad["groups"]["50"]["effects"]["interaction/50/gae_control"] = 999.
        with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(original)
        bad["optimizer_steps"] = 1
        with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)

    def test_aggregation_uses_all_roots_all102_endpoints_and_fixed_correction(self):
        cells = [fixture(r, preflight=False) for r in spec.roots(preflight=False)]
        with patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.np, "quantile", wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(set(summary["endpoints"]), set(spec.ENDPOINTS))
        self.assertEqual(len(summary["endpoints"]), 102)
        self.assertEqual(quantile.call_args.args[1], [.05 / 204, 1 - .05 / 204])
        self.assertEqual(summary["endpoints"]["interaction/50/gae_control"]["ci"], [4., 4.])
        self.assertEqual(summary["cost"]["native_episodes"], 13312)
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1] + [cells[0]], preflight=False)

    def test_full_budget_fresh_seeds_shared_base_and_dynamic_resources(self):
        self.assertEqual(len(spec.VARIANTS), 13)
        self.assertEqual(len(set(spec.ENDPOINTS)), 102)
        b = spec.budget(preflight=False)
        for key, expected in {"calibration_archive_episodes": 4096, "reconstructed_lower_calls": 4915200,
                "native_episodes": 13312, "native_steps": 15974400, "actor_parameter_perturbations": 192,
                "stage73_direction_reproductions": 96, "cross_execution_pair_checks": 512}.items():
            self.assertEqual(b[key] * 8, expected)
        families = []
        for preflight in (True, False):
            for r in spec.roots(preflight=preflight):
                roles, old = spec.seed_roles(r, preflight=preflight), spec.source.seed_roles(r, preflight=preflight)
                self.assertFalse(set(roles["native_evaluation"]).intersection([*old["calibration"], *old["native_evaluation"]]))
                task = task_specification("unit_stage74", r, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            families.append(task["resource_family"])
            q = qualification_task("unit_stage74", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])
        self.assertNotEqual(*families)


if __name__ == "__main__":
    unittest.main()
