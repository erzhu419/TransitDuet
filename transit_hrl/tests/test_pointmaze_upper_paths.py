import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_paths as experiment
from freq_hrl.experiments import pointmaze_upper_execution as production
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_upper_paths_stage75_spec as spec
from scripts.submit_pointmaze_upper_paths_stage75_scheduleurm import task_specification, qualification_task
from test_pointmaze_joint_renewal import DenseTask


def predictor():
    n = len(learned.forecast.forecast_features(np.zeros((64, 2))))
    return {"mean": np.zeros(n), "scale": np.ones(n),
        "weights": np.zeros((n + 1, learned.forecast.spec.FORECAST_STEPS * 2))}


def evaluation(seeds, period=50, horizon=300):
    result = {}
    for mode, effect in zip(spec.MODES, (0., 2., 3., 9.)):
        result[mode] = [{"seed": seed, "mode": mode, "episode_return": float(i) + effect,
            "tracking_squared_error_integral": float(i) - 2 * effect, "episode_length": horizon,
            "policy_seed": seed + 1, "lower_seed": seed + 2, "decision_steps": list(range(0, horizon, period)),
            "upper_proposed_actions": [[0., 0., 0., 0.]] * (horizon // period),
            "upper_calls": horizon // period, "lower_calls": horizon, "network_check": "passed",
            "plan_ols_fits": horizon // period - 1, "plan_ridge_predictions": horizon // period - 1,
            "reference_evaluations": horizon, "actor_context_evaluations": horizon} for i, seed in enumerate(seeds)]
    return result


def fixture(root, *, preflight):
    roles = spec.seed_roles(root, preflight=preflight)
    h = spec.arguments(root, preflight=preflight).horizon
    groups, planning = {}, dict.fromkeys(experiment.PLANNING_KEYS, 0)
    for p in spec.PERIODS:
        ev = evaluation(roles["native_evaluation"], p, h)
        replay = {m: copy.deepcopy(ev[m]) for m in ("R0V0", "R1V1")} if preflight else {}
        groups[str(p)] = {"evaluation": ev, "production_replays": replay,
            "effects": experiment.paired_endpoints(p, ev, roles["native_evaluation"]),
            "pairing": "passed", "source_and_Adam_unchanged": "passed"}
        for rows in [*ev.values(), *replay.values()]:
            for row in rows:
                for key in planning:planning[key] += row[key]
    return {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": spec.budget(preflight=preflight), "groups": groups,
        "native_planning_cost": planning, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0}


class UpperPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_aligned_and_mixed_paths_match_reference_and_velocity_at_every_age(self):
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.001, .002]
        obs = SimpleNamespace(task_measurement=frames[-1].copy())
        history = SimpleNamespace(history=frames.reshape(-1).copy())
        kwargs = dict(observation=obs, history=history, step=63, world_low=-.2 * np.ones(2), world_high=.2 * np.ones(2))
        for period in spec.PERIODS:
            canonical = [production.ExecutedPlan(predictor(), period, .8, m) for m in ("zero_residual", "normal")]
            for plan in canonical:plan.decode(action=np.array([.8, -.7, .6, -.9]), **kwargs)
            for mode in spec.MODES:
                plan = experiment.PathFactorPlan(predictor(), period, .8, mode)
                anchor = plan.decode(action=np.array([.8, -.7, .6, -.9]), **kwargs)
                ref, vel = canonical[int(mode[1])], canonical[int(mode[3])]
                np.testing.assert_array_equal(anchor, ref.points[0])
                for age in range(period):
                    np.testing.assert_array_equal(plan(age=age, observation=obs), ref(age=age))
                    common = dict(age=age, step=63+age, horizon=300)
                    np.testing.assert_array_equal(plan.actor_context(**common), vel.actor_context(**common))
                    np.testing.assert_array_equal(plan.value_context(**common), vel.value_context(**common))
                self.assertEqual((plan.calls, plan.context_calls), (period, period))
                saved = plan.reference_points.copy()
                history.history[:] = 999
                np.testing.assert_array_equal(plan(age=1, observation=obs), saved[1])
                history.history[:] = frames.reshape(-1)
                if mode[1] == "0":self.assertEqual(plan.reference_residual_squared_integral, 0.)
                if mode[3] == "0":self.assertEqual(plan.velocity_residual_squared_integral, 0.)

    def test_fixture_rollouts_match_original_production_without_trace_or_training(self):
        torch.manual_seed(75)
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_value_state_dim=392))
        model = learned.make_model(source)
        args = spec.arguments(310001, preflight=True)
        experiment.init_worker(model.config, args)
        weights = experiment.native.joint.inference_weights(model)
        for period in spec.PERIODS:
            with patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), patch.object(
                    experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))), patch.object(
                    experiment.native.joint, "rollout", wraps=experiment.native.joint.rollout) as rollout:
                ev = {m: [experiment.worker_native((weights, 75095001, m, period, predictor()))] for m in spec.MODES}
                replay = {m: [experiment.native.worker_native((weights, 75095001, arm, period, predictor()))]
                    for m, arm in (("R0V0", "zero_train"), ("R1V1", "joint_ppo"))}
            experiment.check_production(ev, replay, [75095001])
            experiment.paired_endpoints(period, ev, [75095001])
            for call in rollout.call_args_list:
                self.assertFalse(call.kwargs["sample"])
                self.assertFalse(call.kwargs["capture"])
                self.assertTrue(call.kwargs["lower_sample"] and call.kwargs["upper_sample"])
            for row in ev.values():experiment.check_row(row[0], period, args.horizon)

    def test_factorial_signs_and_common_noise_validation(self):
        effects = experiment.paired_endpoints(50, evaluation([1, 2]), [1, 2])
        for name, value in zip(spec.CONTRASTS, (2., 6., 3., 7., 9., 4.)):
            self.assertEqual(effects[f"50/episode_return/{name}"], value)
            self.assertEqual(effects[f"50/tracking_squared_error_integral/{name}"], -2 * value)
        for key, value in (("lower_seed", 999), ("upper_proposed_actions", [[1., 0., 0., 0.]])):
            ev = evaluation([1, 2])
            ev["R1V0"][0][key] = value
            with self.assertRaises(ValueError):experiment.paired_endpoints(50, ev, [1, 2])
        ev = evaluation([1, 2])
        del ev["R0V1"]
        with self.assertRaises(ValueError):experiment.paired_endpoints(50, ev, [1, 2])
        ev = evaluation([1, 2])
        ev["R1V1"].reverse()
        with self.assertRaises(ValueError):experiment.paired_endpoints(50, ev, [1, 2])

    def test_qualification_rejects_budget_production_and_optimization_changes(self):
        original = fixture(310001, preflight=True)
        experiment.qualify(original, preflight=True)
        for key in ("native_episodes", "production_equivalence_checks"):
            bad = copy.deepcopy(original)
            bad["cost"][key] -= 1
            with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(original)
        bad["groups"]["50"]["production_replays"]["R1V1"][0]["episode_return"] += 1.
        with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(original)
        bad["optimizer_steps"] = 1
        with self.assertRaises(ValueError):experiment.qualify(bad, preflight=True)

    def test_all_roots_all24_endpoints_and_fixed_family(self):
        cells = [fixture(r, preflight=False) for r in spec.roots(preflight=False)]
        with patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.np, "quantile", wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(len(summary["endpoints"]), 24)
        self.assertEqual(quantile.call_args.args[1], [.05 / 48, 1-.05 / 48])
        self.assertEqual(summary["endpoints"]["100/episode_return/interaction"]["ci"], [4., 4.])
        self.assertEqual(summary["cost"]["native_episodes"], 2048)
        self.assertEqual(summary["cost"]["native_steps"], 2457600)
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)

    def test_budget_fresh_seeds_and_dynamic_resource_namespaces(self):
        self.assertEqual(spec.budget(preflight=True)["native_episodes"], 48)
        self.assertEqual(spec.budget(preflight=True)["native_steps"], 14400)
        self.assertEqual(spec.budget(preflight=True)["production_equivalence_checks"], 16)
        self.assertEqual(spec.budget(preflight=False)["production_equivalence_checks"], 0)
        families = []
        all_seeds = set()
        for preflight in (True, False):
            for r in spec.roots(preflight=preflight):
                seeds, old = spec.seed_roles(r, preflight=preflight)["native_evaluation"], spec.source.seed_roles(r, preflight=preflight)
                self.assertFalse(set(seeds).intersection([*old["calibration"], *old["native_evaluation"], *all_seeds]))
                all_seeds.update(seeds)
                task = task_specification("unit_stage75", r, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            families.append(task["resource_family"])
            q = qualification_task("unit_stage75", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])
        self.assertNotEqual(*families)


if __name__ == "__main__":
    unittest.main()
