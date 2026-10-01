import copy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_bounded_residual as bounded
from freq_hrl.experiments import pointmaze_calibrated_residual as shared
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_bounded_residual_stage78_spec as spec
from scripts.submit_pointmaze_bounded_residual_stage78_scheduleurm import task_specification, qualification_task
from test_pointmaze_upper_paths import predictor, evaluation
from test_pointmaze_joint_renewal import DenseTask


def calibration():
    responses = {"zero": {"bc_command_mse": .01, "command_change_rms": 0.},
        "original": {"bc_command_mse": .26, "command_change_rms": .5},
        "ratio": {"bc_command_mse": .10, "command_change_rms": .3},
        "bounded": {"bc_command_mse": .012, "command_change_rms": .08}}
    trace = [{"alpha": .2, "response": responses["ratio"]},
        {"alpha": .1, "response": {"bc_command_mse": .04, "command_change_rms": .15}},
        {"alpha": .05, "response": responses["bounded"]}]
    return {"alpha": .05, "ratio_alpha": .2, "target_rms": .1, "solver_trace": trace, "responses": responses,
        "saved_bc_mse": .01, "bounded_to_bc_rmse_ratio": .8, "data_role": "historical_BC_labels_only",
        "bc_mse_reproduction": "passed", "label_plan_checks": "passed", "ratio_reproduction": "passed",
        "constraint": "passed", "envelope": {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}}


def fixture(root, *, preflight):
    roles = spec.seed_roles(root, preflight=preflight)
    h = spec.arguments(root, preflight=preflight).horizon
    groups, planning = {}, dict.fromkeys(shared.paths.PLANNING_KEYS, 0)
    for p in spec.PERIODS:
        ev = dict(zip(spec.MODES, evaluation(roles["native_evaluation"], p, h).values()))
        c = calibration()
        for mode, rows in ev.items():
            for row in rows:row.update(mode=mode, alpha=spec.mode_alphas(c)[mode])
        groups[str(p)] = {"calibration": c, "evaluation": ev, "production_replays": {},
            "effects": shared.paired_endpoints(p, ev, roles["native_evaluation"], protocol=spec),
            "pairing": "passed", "source_and_Adam_unchanged": "passed"}
        for rows in ev.values():
            for row in rows:
                for k in planning:planning[k] += row[k]
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "groups": groups, "native_planning_cost": planning,
        "optimizer_steps": 0, "critic_fits": 0, "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0}
    cell["cost"] = spec.realized_budget(cell, preflight=preflight)
    return cell


class BoundedResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_feasible_halving_does_not_assume_monotonicity(self):
        calls = []
        def evaluate(alpha):
            calls.append(alpha)
            return {"command_change_rms": .12 if alpha == .1 else .02}
        trace = bounded.contract_response(.2, .06, evaluate, {"command_change_rms": .1})
        self.assertEqual(calls, [.1, .05])
        self.assertEqual([r["alpha"] for r in trace], [.2, .1, .05])
        self.assertEqual(len(bounded.contract_response(.2, .2, evaluate, {"command_change_rms": .1})), 1)

    def test_constraint_rejects_changed_target_trace_or_nonfeasible_response(self):
        bounded.check_calibration(calibration())
        mutations = (lambda c: c.update(target_rms=.2), lambda c: c.update(alpha=.1),
            lambda c: c["solver_trace"][1].update(alpha=.15),
            lambda c: c["responses"]["bounded"].update(command_change_rms=.11),
            lambda c: c["solver_trace"][1]["response"].update(command_change_rms=.09))
        for mutate in mutations:
            c = calibration();mutate(c)
            with self.assertRaises(ValueError):bounded.check_calibration(c)

    def test_actual_actor_calibration_reproduces_ratio_and_counts_each_pass(self):
        torch.manual_seed(78)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=392,
            upper_action_dim=4, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False))
        states = np.zeros((8, 392), dtype=np.float32)
        achieved = np.zeros((8, 2), dtype=np.float32)
        base = np.zeros((2, 5, 2), dtype=np.float32)
        original = base.copy();original[:, 1:] = np.arange(1, 5)[None, :, None]*.03
        target = shared.commands(model.lower_actor, states)+.03
        saved = shared.response(shared.commands(model.lower_actor, states), shared.commands(model.lower_actor, states), target)["bc_command_mse"]
        raw = {k: np.zeros(1) for k in shared.support.RAW_KEYS}
        raw.update(achieved_before=achieved, action=target)
        loader = MagicMock();loader.__enter__.return_value = raw
        args, roles = SimpleNamespace(horizon=8), {"calibration_labels": [1]}
        snapshot = copy.deepcopy(model.state_dict())
        with patch.object(shared.np, "load", return_value=loader), patch.object(shared.support, "label_states", return_value=states), patch.object(
                shared, "historical_curves", return_value=(base, original, 2)), patch.object(spec, "arguments", return_value=args):
            prior_cost = dict.fromkeys(shared.spec.budget(preflight=True), 0)
            prior = shared.calibrate(310011, 50, model=model, predictor=predictor(), args=args, roles=roles,
                bounds=None, old={"bc_mse_reproduction": {"saved": saved}, "calibration": {}}, cost=prior_cost, preflight=False)
            cost = dict.fromkeys(spec.budget(preflight=True), 0)
            result = bounded.calibrate(310011, 50, model=model, predictor=predictor(), args=args, roles=roles,
                bounds=None, old={"calibration": prior}, cost=cost, preflight=True)
        self.assertEqual(result["ratio_alpha"], prior["alpha"])
        self.assertEqual(result["responses"]["ratio"], prior["responses"]["calibrated"])
        self.assertLessEqual(result["bounded_to_bc_rmse_ratio"], 1.)
        self.assertEqual(cost["constraint_response_evaluations"], len(result["solver_trace"]))
        self.assertEqual(cost["offline_actor_rows"], (2+len(result["solver_trace"]))*8)
        self.assertEqual(cost["offline_actor_forward_batches"], 2+len(result["solver_trace"]))
        self.assertEqual(cost["native_steps"], 0)
        shared.support.assert_frozen(model, snapshot)

    def test_four_native_modes_keep_common_noise_and_do_not_collect_training(self):
        torch.manual_seed(78)
        model = learned.make_model(FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_value_state_dim=392)))
        args = spec.arguments(310011, preflight=True)
        shared.init_worker(model.config, args)
        weights, c = shared.paths.native.joint.inference_weights(model), calibration()
        for p in spec.PERIODS:
            with patch.object(shared.paths.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), patch.object(
                    shared.paths.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), patch.object(
                    shared.paths.native.joint, "rollout", wraps=shared.paths.native.joint.rollout) as rollout:
                ev = {m: [shared.worker_native((weights, 78095001, m, p, predictor(), spec.mode_alphas(c)[m], c["envelope"]))] for m in spec.MODES}
            effects = shared.paired_endpoints(p, ev, [78095001], protocol=spec)
            self.assertEqual(len(effects), 12)
            for call in rollout.call_args_list:
                self.assertFalse(call.kwargs["sample"] or call.kwargs["capture"])
                self.assertTrue(call.kwargs["upper_sample"] and call.kwargs["lower_sample"])
            ev["bounded"][0]["upper_proposed_actions"][0][0] += 1
            with self.assertRaises(ValueError):shared.paired_endpoints(p, ev, [78095001], protocol=spec)

    def test_complete_roots_dynamic_offline_work_and_all24_endpoints(self):
        c = fixture(310011, preflight=True)
        shared.qualify(c, preflight=True, protocol=spec, calibration_check=bounded.check_calibration)
        self.assertEqual(c["cost"]["native_episodes"], 32)
        self.assertEqual(c["cost"]["label_state_rows"], 19200)
        c["cost"]["offline_actor_rows"] -= 1
        with self.assertRaises(ValueError):shared.qualify(c, preflight=True, protocol=spec, calibration_check=bounded.check_calibration)
        cells = [fixture(r, preflight=False) for r in spec.roots(preflight=False)]
        summary = shared.aggregate(cells, preflight=False, protocol=spec, calibration_check=bounded.check_calibration)
        self.assertEqual(len(summary["endpoints"]), 24)
        self.assertEqual(summary["cost"]["native_episodes"], 2048)
        self.assertEqual(summary["cost"]["native_steps"], 2457600)
        self.assertEqual(summary["endpoints"]["50/episode_return/bounded_minus_ratio"]["mean"], 6.)
        self.assertEqual(summary["endpoints"]["50/episode_return/bounded_minus_ratio"]["effect"], "positive")
        with self.assertRaises(ValueError):shared.aggregate(cells[:-1], preflight=False, protocol=spec, calibration_check=bounded.check_calibration)

    def test_full_source_even_in_preflight_and_dynamic_scheduler(self):
        for preflight in (True, False):
            self.assertFalse(spec.source_preflight(preflight))
            for r in spec.roots(preflight=preflight):
                roles = spec.seed_roles(r, preflight=preflight)
                prior = spec.source.seed_roles(r, preflight=False)
                self.assertEqual(roles["calibration_labels"], prior["calibration_labels"])
                self.assertFalse(set(roles["native_evaluation"]) & set(prior["native_evaluation"]+prior["calibration_labels"]))
            task = task_specification("stage78_test", 310011, preflight=preflight)
            self.assertEqual(task["cpu"], spec.options(preflight=preflight)["workers"]+1)
            self.assertEqual(task["ram_mb"], 3072 if preflight else 8192)
            self.assertIsNone(task["require_node"])
            self.assertEqual(set(task["allowed_nodes"]), {f"node{i:03d}" for i in range(1, 7)})
            self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertFalse(any("replicate_310001" in p for p in task.get("wait_for_files", [])))
            q = qualification_task("stage78_test", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))


if __name__ == "__main__":
    unittest.main()
