import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_velocity_support as experiment
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.experiments import pointmaze_upper_paths as paths
from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_velocity_support_stage76_spec as spec
from scripts.submit_pointmaze_velocity_support_stage76_scheduleurm import task_specification, qualification_task
from test_pointmaze_upper_paths import predictor
from test_pointmaze_joint_renewal import DenseTask


def fixture(root, *, preflight):
    roles, args = spec.seed_roles(root, preflight=preflight), spec.arguments(root, preflight=preflight)
    groups = {}
    for p in spec.PERIODS:
        coverage = {v: {"rows": len(roles["native_plan_replay"]) * args.horizon,
            "outside_label_axis_range_rate": .1 if v == "base" else .4,
            "above_label_speed_q99_rate": .05 if v == "base" else .2} for v in ("base", "residual")}
        groups[str(p)] = {"calibration": {"rows": len(roles["calibration_labels"]) * args.horizon, "data_role": "historical_BC_labels_only"},
            "label_plan_check": "passed", "native_plan_frame_check": "passed", "source_and_Adam_unchanged": "passed",
            "bc_mse_reproduction": {"saved": .001, "observed": .001, "status": "passed"},
            "actor_responses": dict.fromkeys(spec.VARIANTS, {}), "native_coverage": coverage, "effects": experiment.effects(p, coverage)}
    return {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "groups": groups, "cost": spec.budget(preflight=preflight)}


class VelocitySupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_curve_replay_matches_production_and_freezes_each_observed_prefix(self):
        m = np.zeros((300, 6), dtype=np.float32)
        m[:, :2] = np.arange(300)[:, None] * [.001, .002]
        bounds = (-2 * np.ones(2), 2 * np.ones(2))
        for p in spec.PERIODS:
            actions = np.tile([.3, -.4, .2, -.1], (300 // p, 1)).astype(np.float32)
            br, r, bv, v, cost = experiment.replay_plan(m, actions, predictor=predictor(), period=p, scale=.5, bounds=bounds)
            actual = paths.PathFactorPlan(predictor(), p, .5, "R1V1")
            for i, start in enumerate(range(0, 300, p)):
                from types import SimpleNamespace
                actual.decode(action=actions[i], observation=None, history=SimpleNamespace(history=m[max(0, start-63):start+1].reshape(-1)),
                    step=start, world_low=bounds[0], world_high=bounds[1])
                for age in range(p):
                    obs = SimpleNamespace(task_measurement=m[start+age])
                    np.testing.assert_array_equal(r[start+age], actual(age=age, observation=obs))
                    np.testing.assert_array_equal(v[start+age], actual.actor_context(age=age, step=start+age, horizon=300))
            row = {k:getattr(actual,k) for k in ("reference_target_squared_error_integral", "reference_residual_squared_integral", "velocity_residual_squared_integral")}
            experiment.check_native_frame(br, r, bv, v, m, row)
            row["velocity_residual_squared_integral"] += .01
            with self.assertRaises(AssertionError):experiment.check_native_frame(br, r, bv, v, m, row)
            changed = m.copy();changed[1:p, :2] += 1.
            replay = experiment.replay_plan(changed, actions, predictor=predictor(), period=p, scale=.5, bounds=bounds)
            np.testing.assert_array_equal(replay[1][:p], r[:p])
            self.assertEqual(cost["plan_decodes"], 300 // p)

    def test_label_states_are_the_exact_training_inputs(self):
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_value_state_dim=392))
        model, args = learned.make_model(source), spec.arguments(310001, preflight=True)
        plan = learned.ResidualPlan(predictor(), 50, args.maximum_subgoal_delta)
        with patch.object(experiment.native.joint, "_make_task", return_value=DenseTask()), patch.object(
                experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
            batch, _, raw = experiment.native.joint.rollout(model, args, "fixed50", seed=1, sample=True, capture=True,
                upper_sample=False, lower_sample=False, lower_credit="task_option", upper_plan_decoder=plan.decode,
                lower_reference_builder=plan, lower_actor_context_builder=plan.actor_context, lower_value_context_builder=plan.value_context)
        np.testing.assert_array_equal(experiment.label_states(raw, args), batch.lower.state)

    def test_support_is_fit_only_from_labels_and_profiles_have_correct_signs(self):
        v = np.array([[-1., 0.], [1., 0.], [0., 1.], [0., -1.]])
        s = experiment.fit_support(v, np.ones((4, 2)))
        a = experiment.profile(np.array([[0., 0.], [2., 0.], [0., -2.], [1., 0.]]), s)
        self.assertEqual(s["velocity_speed_q99"], 1.)
        self.assertAlmostEqual(s["position_error_q99"], np.sqrt(2.))
        self.assertEqual(a["outside_label_axis_range_rate"], .5)
        self.assertEqual(a["above_label_speed_q99_rate"], .5)
        self.assertEqual(s["data_role"], "historical_BC_labels_only")

    def test_actor_counterfactuals_hold_other_columns_and_do_not_change_parameters(self):
        actor = GaussianActor(392, 2, 8, -.5)
        actor.net = torch.nn.Linear(392, 2, bias=False)
        with torch.no_grad():
            actor.net.weight.zero_();actor.net.weight[:, -2:] = torch.eye(2)
        before = copy.deepcopy(actor.state_dict())
        x = np.zeros((10, 392), dtype=np.float32);x[:, -2:] = [1., 2.]
        y = np.tanh(x[:, -2:])
        totals, cost = experiment.actor_responses(actor, x, y, np.tile([-1., -2.], (10, 1)))
        response = experiment.response_summary(totals, len(x))
        self.assertLess(response["as_label"]["bc_command_mse"], 1e-12)
        self.assertEqual(response["as_label"]["conditional_gaussian_kl_mean"], 0.)
        self.assertGreater(response["zero_velocity"]["command_change_rms"], .5)
        self.assertGreater(response["sampled_velocity"]["command_change_rms"], 1.)
        self.assertEqual(cost, {"offline_actor_rows": 30, "offline_actor_forward_batches": 3})
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        np.testing.assert_array_equal(x[:, -2:], np.tile([1., 2.], (10, 1)))

    def test_qualification_rejects_native_steps_mse_and_coverage_accounting_changes(self):
        c = fixture(310001, preflight=True)
        experiment.qualify(c, preflight=True)
        for mutation in ("native_steps", "mse", "coverage"):
            bad = copy.deepcopy(c)
            if mutation == "native_steps":bad["cost"]["native_steps"] = 1
            if mutation == "mse":bad["groups"]["50"]["bc_mse_reproduction"]["observed"] += .01
            if mutation == "coverage":bad["groups"]["50"]["native_coverage"]["base"]["rows"] -= 1
            with self.assertRaises((ValueError, AssertionError)):experiment.qualify(bad, preflight=True)

    def test_four_endpoints_all_roots_and_small_dynamic_scheduler_resources(self):
        cells = [fixture(r, preflight=False) for r in spec.roots(preflight=False)]
        with patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.np, "quantile", wraps=np.quantile) as q:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(len(summary["endpoints"]), 4)
        self.assertEqual(q.call_args.args[1], [.05/8, 1-.05/8])
        self.assertEqual(summary["cost"]["label_state_rows"], 153600)
        self.assertEqual(summary["cost"]["native_plan_frame_checks"], 512)
        self.assertEqual(summary["cost"]["replayed_velocity_rows"], 1536000)
        self.assertEqual(summary["cost"]["offline_actor_forward_batches"], 768)
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)
        for preflight in (True, False):
            task = task_specification("unit_stage76", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 2048))
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(task.get("require_node"))
            self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertEqual(len(qualification_task("unit_stage76", preflight=preflight)["wait_for_files"]), len(spec.roots(preflight=preflight)))


if __name__ == "__main__":unittest.main()
