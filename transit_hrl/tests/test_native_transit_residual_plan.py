import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core.prefix_cost_credit import PrefixCostCredit
from freq_hrl.domains.transit.native_residual_plan import NativeResidualPlan, prefix_service_cost, residual_basis
from scripts import run_native_transit_residual_plan_stage157 as spec
from scripts.analyze_native_transit_residual_plan_stage157 import summarize
from scripts.submit_native_transit_residual_plan_stage157_scheduleurm import task_specification
from tests import test_native_transit_service_plan as plan_tests


class NativeResidualPlanTest(unittest.TestCase):
    def test_prefix_credit_telescopes_over_variable_durations_and_tail(self):
        ledger = PrefixCostCredit()
        ledger.begin([0], [.1, -.1], 5, 0)
        ledger.begin([1], [.2, -.2], 6, 60)
        ledger.begin([2], [.3, -.3], 4, 1000)
        summary = ledger.finish(3, 61380)
        self.assertEqual(summary["reward_sum"], 200)
        self.assertEqual(summary["duration_s"], 61380)
        self.assertEqual([t["duration_s"] for t in ledger.transitions], [60, 940, 60380])
        self.assertEqual([t["done"] for t in ledger.transitions], [False, False, True])
        np.testing.assert_array_equal(ledger.transitions[0]["next_state"], [1])
        with self.assertRaises(ValueError):
            ledger.finish(3, 61380)

    def environment(self, action):
        env, _ = plan_tests.ServicePlanTest().environment("causal_forecast")
        env._build_upper_state_v2 = lambda trip: np.zeros(16, dtype=np.float32)
        env.stations, env._completed_trip_ids, env._peak_concurrent = [], set(), 0
        env.protocol = SimpleNamespace(evaluation_end_time_s=2700)
        env.headway_events = SimpleNamespace(summary=lambda: {"headway_cv": 0.0})
        plan = NativeResidualPlan(env, lambda state: np.asarray(action, dtype=np.float32))
        env._upper_policy_callback = plan
        return env, plan

    def execute(self, action):
        env, plan = self.environment(action)
        for tick in range(2700):
            env.current_time = tick
            env._dispatch_due_trips()
        return env, plan

    def test_zero_residual_is_exact_forecast_and_actual_decisions_not_trip_queries(self):
        env, plan = self.execute([0, 0])
        reference_env, reference = plan_tests.ServicePlanTest().execute("causal_forecast")
        self.assertEqual(plan.blocks, reference.blocks)
        self.assertEqual([t._actual_launch_time for t in env.timetables],
                         [t._actual_launch_time for t in reference_env.timetables])
        self.assertEqual(len(plan.decisions), 2)
        self.assertEqual(len(plan.queries), 12)
        self.assertTrue(all(d["state"].shape == (34,) for d in plan.decisions))
        env._measurement_details = {"restricted_wait_horizon_min": 0, "peak_fleet": 0,
            "headway_cv": 0, "passenger_unserved_rate": 0, "trip_completion_rate": 0}
        summary = plan.finish({"service_cost_restricted": 5})
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["reward_sum"], 0)
        self.assertEqual(summary["duration_s"], 2699 - 180)

    def test_residual_changes_executable_plan_not_endpoints_or_event_budget(self):
        _, zero = self.execute([0, 0])
        env, plan = self.execute([1, -.5])
        self.assertNotEqual(zero.blocks[0]["planned_s"], plan.blocks[0]["planned_s"])
        self.assertEqual(plan.summarize()["endpoint_error_max_s"], 0)
        self.assertEqual(plan.summarize()["actual_target_error_abs_mean_s"], 0)
        for intervals in (2, 4, 5):
            basis = residual_basis(intervals)
            np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-14)
            self.assertTrue(np.all(np.abs(basis) <= 1))
        for bad in ([2, 0], [float("nan"), 0], [0]):
            env, _ = self.environment(bad)
            env.current_time = 180
            with self.assertRaises(ValueError):
                env._dispatch_due_trips()

    def test_prefix_cost_censors_at_current_clock_not_future_evaluation_end(self):
        env, _ = self.environment([0, 0])
        env.current_time = 60
        env.stations = [SimpleNamespace(total_passenger=[SimpleNamespace(appear_time=0, boarding_time=None)])]
        # wait=1min/10, unserved=5, incomplete=5; NOT a 2700-second wait.
        self.assertAlmostEqual(prefix_service_cost(env), 10.1)
        env.protocol.evaluation_end_time_s = 1000000
        self.assertAlmostEqual(prefix_service_cost(env), 10.1)

    def test_terminal_credit_uses_cached_measurements_after_native_passenger_cleanup(self):
        env, plan = self.environment([0, 0])
        env.current_time = 2700
        env._measurement_details = {"restricted_wait_horizon_min": 10, "peak_fleet": 14,
            "headway_cv": .2, "passenger_unserved_rate": .1, "trip_completion_rate": 1}
        plan.credit.begin(np.zeros(34), [0, 0], 5, 0)
        expected = 1 + 4 / 12 + .2 + .5
        self.assertNotAlmostEqual(prefix_service_cost(env), expected)
        summary = plan.finish({"service_cost_restricted": round(expected, 6)})
        self.assertAlmostEqual(summary["final_cost"], expected)
        self.assertAlmostEqual(summary["reward_sum"], 100 * (5 - expected))

    def test_native_constructor_does_not_reset_upper_exploration_rng(self):
        def constructor(*args, **kwargs):
            torch.rand(13)
            return SimpleNamespace(load_checkpoint=lambda *a, **kw: None,
                upper_trainer=SimpleNamespace(update=lambda: None), lower_trainer=SimpleNamespace(update=lambda: None))
        module = SimpleNamespace(TransitDuetV2Runner=constructor,
            load_config=lambda path: {k: {} for k in ("frequency", "env", "coupling", "upper", "lower")})
        torch.manual_seed(113)
        before = torch.random.get_rng_state().clone()
        with patch.dict(sys.modules, {"runner_v3": module}):
            spec.make_runner(397, "low_noise", Path("unused"), Path("unused"))
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_short_training_runs_shared_sac_actor_and_critic(self):
        def fake_episode(root, scenario, scene, raw, checkpoint, action_fn, **kw):
            ledger = PrefixCostCredit()
            decisions = []
            for index in range(4):
                state = np.full(34, index / 4, dtype=np.float32)
                action = action_fn(state)
                ledger.begin(state, action, 5 - index / 4, index * 1000)
                decisions.append({"state": state, "action": action})
            credit = ledger.finish(3, 5400)
            row = {k: 1. for k in spec.source.source_spec.authority.routing.METRICS}
            row.update(service_cost_restricted=3, ep=300, N_fleet=12, simulation_end_time_s=5400,
                done_reason="evaluation_horizon", passengers_generated=100, passengers_unserved=0,
                trips_completed=24, ep_steps=100)
            return row, SimpleNamespace(credit=ledger, decisions=decisions), credit, {"endpoint_error_max_s": 0}
        with patch.object(spec, "episode", side_effect=fake_episode):
            agent, result = spec.train(397, Path("unused"), Path("unused"), preflight=True)
        self.assertEqual(agent.update_step, 2)
        self.assertEqual(result["transitions"], 8)
        self.assertGreater(result["actor_change_max_abs"], 0)
        self.assertGreater(result["learning_mean"]["critic_loss"], 0)
        self.assertEqual(len(result["constant_action"]), 2)

    def cells(self):
        def credit(cost):
            return {"decisions": 44, "terminal_transitions": 1, "duration_s": 61380,
                "initial_cost": 5, "final_cost": cost, "reward_sum": 100 * (5 - cost)}
        cells = {}
        for root in spec.ROOTS:
            rows = []
            for index, condition in enumerate(spec.CONDITIONS):
                for scenario in spec.contract()["scenarios"]:
                    for scene in spec.source.source_spec.scene_seeds(spec.LOWER_ROOT[root], scenario, preflight=False):
                        rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                            **{k: float(index + 1) for k in spec.source.source_spec.authority.routing.METRICS},
                            "N_fleet": 12, "simulation_end_time_s": 61380, "passengers_generated": 100,
                            "execution": {"endpoint_error_max_s": 0}, "source_control_reproduced": True,
                            "credit": credit(index + 1), "residual_action_mean": [.1, .2], "residual_action_std": [0, 0]})
            curve = {"service_cost_restricted": 1, "credit": credit(1)}
            training = {"updates": 5500, "actor_change_max_abs": .1, "learning_mean": {"critic_loss": 1},
                "training_scene_seeds": [700000000 + root * 1000 + ep for ep in range(120)],
                "training_curve": [curve], "constant_action": [.1, .2]}
            short = copy.deepcopy(training)
            short["updates"] = 2
            short["training_curve"][0]["credit"]["duration_s"] = 5400
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": root, "lower_root": spec.LOWER_ROOT[root], "software_qualified": True,
                "native_training_updates": 0, "native_steps": 200 * 61380,
                "worker_preflight_native_steps": 3 * 61380 + 2 * 5400,
                "qualification": [{"condition": c, "reproduced": True} for c in
                    ("source_baseline", "nominal_plan", "causal_forecast")],
                "training": training, "preflight_learning": short, "evaluation": rows}
        return cells

    def test_analysis_requires_common_credit_source_controls_and_actual_training(self):
        cells = self.cells()
        result = summarize(cells)
        self.assertEqual(result["native_steps"], 400 * 61380)
        self.assertIn("frozen_lower", result["stage"])
        self.assertEqual(result["learned_minus_control"]["forecast"][0]["service_cost_restricted"], -1)
        for failure in ("credit", "source", "constant", "updates", "duplicate"):
            broken = copy.deepcopy(cells)
            c = broken[397]
            r = next(r for r in c["evaluation"] if r["condition"] == "constant_residual")
            if failure == "credit":
                r["credit"]["reward_sum"] += 1
            elif failure == "source":
                next(r for r in c["evaluation"] if r["condition"] == "forecast")["source_control_reproduced"] = False
            elif failure == "constant":
                r["residual_action_mean"][0] += .1
            elif failure == "updates":
                c["training"]["updates"] -= 1
            else:
                c["evaluation"].append(copy.deepcopy(r))
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                summarize(broken)

    def test_scheduler_is_unpinned_code_only(self):
        task = task_specification("test_residual", 397)
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIsNone(task.get("require_node"))
        self.assertEqual([p.split("/")[-1] for p in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])


if __name__ == "__main__":
    unittest.main()
