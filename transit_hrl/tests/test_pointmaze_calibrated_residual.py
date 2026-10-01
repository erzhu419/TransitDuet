import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_calibrated_residual as experiment
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_calibrated_residual_stage77_spec as spec
from scripts.submit_pointmaze_calibrated_residual_stage77_scheduleurm import task_specification, qualification_task
from test_pointmaze_upper_paths import predictor, evaluation
from test_pointmaze_joint_renewal import DenseTask


def fixture(root, *, preflight):
    roles = spec.seed_roles(root, preflight=preflight)
    h = spec.arguments(root, preflight=preflight).horizon
    groups, planning = {}, dict.fromkeys(experiment.paths.PLANNING_KEYS, 0)
    for p in spec.PERIODS:
        source = evaluation(roles["native_evaluation"], p, h)
        ev = {m: source[old] for m, old in zip(spec.MODES, ("R0V0", "R1V1", "R1V0"))}
        for mode, rows in ev.items():
            for row in rows:row.update(mode=mode, alpha={"zero": 0., "original": 1., "calibrated": .2}[mode])
        replay = {old: copy.deepcopy(ev[m]) for old, m in (("R0V0", "zero"), ("R1V1", "original"))} if preflight else {}
        calibration = {"alpha": .2, "data_role": "historical_BC_labels_only", "saved_bc_mse": .01,
            "label_plan_checks": "passed", "bc_mse_reproduction": "passed",
            "responses": {"zero": {"bc_command_mse": .01}, "original": {"command_change_rms": .5}}}
        groups[str(p)] = {"calibration": calibration, "evaluation": ev, "production_replays": replay,
            "effects": experiment.paired_endpoints(p, ev, roles["native_evaluation"]),
            "pairing": "passed", "source_and_Adam_unchanged": "passed"}
        for rows in [*ev.values(), *replay.values()]:
            for row in rows:
                for k in planning:planning[k] += row[k]
    return {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": spec.budget(preflight=preflight), "groups": groups,
        "native_planning_cost": planning, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0}


class CalibratedResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_curve_endpoints_coherence_anchor_and_clipped_bounds(self):
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.001, .002]
        kwargs = dict(action=np.array([.8, -.7, .6, -.9]),
            observation=SimpleNamespace(task_measurement=frames[-1]), history=SimpleNamespace(history=frames.reshape(-1)),
            step=63, world_low=-.2*np.ones(2), world_high=.2*np.ones(2))
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        for p in spec.PERIODS:
            original = learned.ResidualPlan(predictor(), p, .8)
            original.decode(**kwargs)
            for alpha in (0., .2, 1.):
                plan = experiment.CalibratedPlan(predictor(), p, .8, alpha, envelope)
                plan.decode(**kwargs)
                points = experiment.blend(original.base_points, original.points, alpha)
                np.testing.assert_array_equal(plan.points, points)
                np.testing.assert_array_equal(plan.points[0], original.base_points[0])
                self.assertTrue(np.all((plan.points >= -.2) & (plan.points <= .2)))
                for age in range(p):
                    np.testing.assert_array_equal(plan(age=age, observation=kwargs["observation"]), points[age])
                    context = dict(age=age, step=age + 63, horizon=300)
                    velocity = (points[age+1]-points[age])/.01
                    np.testing.assert_array_equal(plan.actor_context(**context), velocity)
                    np.testing.assert_array_equal(plan.value_context(**context)[:2], velocity)
                expected = float(np.square(points.astype(np.float64)-original.base_points).sum())
                self.assertEqual(plan.executed_delta_squared_sum, expected)

    def test_joint_state_changes_only_reference_error_and_velocity(self):
        states = np.arange(8*392, dtype=np.float32).reshape(8, 392)
        achieved = np.ones((8, 2), dtype=np.float32)
        points = np.arange(2*5*2, dtype=np.float32).reshape(2, 5, 2)/10
        result = experiment.curve_states(states, achieved, points)
        np.testing.assert_array_equal(result[:, :4], states[:, :4])
        np.testing.assert_array_equal(result[:, 6:390], states[:, 6:390])
        np.testing.assert_array_equal(result[:, 4:6], points[:, :-1].reshape(8, 2)-achieved)
        np.testing.assert_array_equal(result[:, -2:], (np.diff(points, axis=1)/.01).reshape(8, 2))

    def test_calibration_uses_command_errors_and_no_reward_or_scale_search(self):
        self.assertEqual(experiment.calibration_alpha(.01, .5), .2)
        self.assertEqual(experiment.calibration_alpha(.04, .1), 1.)
        self.assertEqual(experiment.calibration_alpha(.01, 0.), 1.)
        torch.manual_seed(77)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=392,
            upper_action_dim=4, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False))
        states = np.zeros((8, 392), dtype=np.float32)
        achieved = np.zeros((8, 2), dtype=np.float32)
        base = np.zeros((2, 5, 2), dtype=np.float32)
        original = base.copy();original[:, 1:] = np.arange(1, 5)[None, :, None]*.02
        bc_command = experiment.commands(model.lower_actor, states)
        target = bc_command + .03
        expected_mse = experiment.response(bc_command, bc_command, target)["bc_command_mse"]
        raw = {k: np.zeros(1) for k in experiment.support.RAW_KEYS}
        raw.update(achieved_before=achieved, action=target)
        loader = unittest.mock.MagicMock();loader.__enter__.return_value = raw
        cost = dict.fromkeys(spec.budget(preflight=True), 0)
        snapshot = copy.deepcopy(model.state_dict())
        with patch.object(experiment.np, "load", return_value=loader), patch.object(experiment.support, "label_states", return_value=states), patch.object(
                experiment, "historical_curves", return_value=(base, original, 2)):
            result = experiment.calibrate(310001, 50, model=model, predictor=predictor(),
                args=SimpleNamespace(horizon=8), roles={"calibration_labels": [1]}, bounds=None,
                old={"bc_mse_reproduction": {"saved": expected_mse}, "calibration": {}}, cost=cost, preflight=True)
        self.assertEqual(result["alpha"], experiment.calibration_alpha(expected_mse, result["responses"]["original"]["command_change_rms"]))
        self.assertEqual(cost["offline_actor_rows"], 24)
        self.assertEqual(cost["native_steps"], 0)
        experiment.support.assert_frozen(model, snapshot)

    def test_native_pairing_and_original_execution_equivalence(self):
        torch.manual_seed(77)
        model = learned.make_model(FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_value_state_dim=392)))
        args = spec.arguments(310001, preflight=True)
        experiment.init_worker(model.config, args)
        weights = experiment.paths.native.joint.inference_weights(model)
        calibration = {"alpha": .2, "envelope": {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}}
        for p in spec.PERIODS:
            with patch.object(experiment.paths.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), patch.object(
                    experiment.paths.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), patch.object(
                    experiment.paths.native.joint, "rollout", wraps=experiment.paths.native.joint.rollout) as rollout:
                ev = {m: [experiment.worker_native((weights, 77095001, m, p, predictor(), calibration))] for m in spec.MODES}
                replay = {m: [experiment.paths.native.worker_native((weights, 77095001, arm, p, predictor()))]
                    for m, arm in (("R0V0", "zero_train"), ("R1V1", "joint_ppo"))}
            experiment.production_check(ev, replay, [77095001])
            experiment.paired_endpoints(p, ev, [77095001])
            for call in rollout.call_args_list:
                self.assertFalse(call.kwargs["sample"])
                self.assertFalse(call.kwargs["capture"])
                self.assertTrue(call.kwargs["upper_sample"] and call.kwargs["lower_sample"])
            for rows in ev.values():experiment.paths.check_row(rows[0], p, args.horizon)
            ev["calibrated"][0]["lower_seed"] += 1
            with self.assertRaises(ValueError):experiment.paired_endpoints(p, ev, [77095001])

    def test_all_roots_budget_and_twelve_corrected_contrasts(self):
        cell = fixture(310001, preflight=True)
        experiment.qualify(cell, preflight=True)
        for mutate in (lambda c: c["cost"].update(native_steps=0), lambda c: c.update(optimizer_steps=1),
                lambda c: c["groups"]["50"]["calibration"].update(alpha=.3),
                lambda c: c["groups"]["50"]["evaluation"]["calibrated"][0].update(alpha=.3)):
            broken = copy.deepcopy(cell);mutate(broken)
            with self.assertRaises(ValueError):experiment.qualify(broken, preflight=True)
        cells = [fixture(r, preflight=False) for r in spec.roots(preflight=False)]
        result = experiment.aggregate(cells, preflight=False)
        self.assertEqual(len(result["endpoints"]), 12)
        self.assertEqual(result["cost"]["native_episodes"], 1536)
        self.assertEqual(result["cost"]["native_steps"], 1843200)
        self.assertEqual(result["endpoints"]["50/episode_return/calibrated_minus_original"]["mean"], -7.)
        self.assertEqual(result["endpoints"]["50/episode_return/calibrated_minus_original"]["effect"], "negative")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)

    def test_fresh_seeds_and_dynamic_worker_resources(self):
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                previous = spec.source.seed_roles(root, preflight=preflight)
                self.assertFalse(set(roles["native_evaluation"]) & set(previous["native_plan_replay"]+roles["calibration_labels"]))
            task = task_specification("stage77_test", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertEqual(task["cpu"], spec.options(preflight=preflight)["workers"]+1)
            self.assertEqual(task["ram_mb"], 3072 if preflight else 8192)
            self.assertIsNone(task["require_node"])
            self.assertEqual(set(task["allowed_nodes"]), {f"node{i:03d}" for i in range(1, 7)})
            q = qualification_task("stage77_test", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))


if __name__ == "__main__":
    unittest.main()
