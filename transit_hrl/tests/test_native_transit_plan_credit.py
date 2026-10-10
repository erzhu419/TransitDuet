import copy
import unittest

import numpy as np
import torch

from scripts import run_native_transit_plan_credit_stage158 as spec
from scripts.analyze_native_transit_plan_credit_stage158 import summarize
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification


class ToyAgent:
    device = torch.device("cpu")

    def act(self, state, *, sample):
        return np.asarray([.3, -.5], dtype=np.float32)

    def critic(self, state, action):
        return action[:, :1], action[:, :1] + .2


def cost_trace(final=1):
    costs = np.linspace(5, final, 45)
    return [{"state": [i / 44] * 34, "action": [.3, -.5], "time_s": i * 1000,
        "cost_before": float(costs[i]), "cost_after": float(costs[i + 1]),
        "reward": 100 * float(costs[i] - costs[i + 1]),
        "duration_s": 1000 if i < 43 else 61380 - 43000, "done": i == 43} for i in range(44)]


def control(final=1):
    row = {k: .1 for k in spec.source.source.source_spec.authority.routing.METRICS}
    row.update(service_cost_restricted=round(final, 6), simulation_end_time_s=61380,
        N_fleet=12, passengers_generated=100,
        credit={"decisions": 44, "terminal_transitions": 1, "duration_s": 61380,
            "initial_cost": 5, "final_cost": final, "reward_sum": 100 * (5 - final)},
        execution={"endpoint_error_max_s": 0}, trace=cost_trace(final))
    return row


class PlanCreditDiagnosticTest(unittest.TestCase):
    def test_intervention_changes_one_action_and_requires_matching_prefix(self):
        agent = ToyAgent()
        ref = [{"state": np.full(34, i, dtype=np.float32).tolist(),
                "action": agent.act(None, sample=False).tolist()} for i in range(4)]
        callback = spec.SinglePlanIntervention(agent, ref, 2, [0, 0])
        actions = [callback(np.full(34, i if i < 3 else 10, dtype=np.float32)) for i in range(4)]
        np.testing.assert_array_equal(actions[2], [0, 0])
        for i in (0, 1, 3):
            np.testing.assert_array_equal(actions[i], agent.act(None, sample=False))
        self.assertEqual(callback.calls, 4)
        with self.assertRaises(RuntimeError):
            spec.SinglePlanIntervention(agent, ref, 2, [0, 0])(np.ones(34, dtype=np.float32))

    def test_pairing_cancels_common_progress_and_preserves_terminal_objective(self):
        def trace(costs):
            return [{"time_s": i * 1000, "cost_before": a, "cost_after": b, "reward": 100 * (a - b)}
                    for i, (a, b) in enumerate(zip(costs, costs[1:]))]
        left, right = trace([5, 8, 7, 1]), trace([5, 8.1, 7.1, 1.2])
        result = spec.paired_credit(left, right)
        self.assertAlmostEqual(result["paired_reward_sum"], 20)
        self.assertLess(result["paired"]["std"], result["raw_learned"]["std"])
        right[1]["time_s"] += 1
        with self.assertRaises(ValueError):
            spec.paired_credit(left, right)

    def test_probe_uses_precise_terminal_cost_not_rounded_row(self):
        left, right = control(1.0000004), control(1.0000006)
        right["trace"][8]["cost_before"] = left["trace"][8]["cost_before"]
        right["trace"][8]["action"] = [0, 0]
        result = spec.probe_result(ToyAgent(), left, right, 8, [0, 0])
        self.assertAlmostEqual(result["physical_return_delta"], -.00002)
        self.assertAlmostEqual(result["critic_margin"], -.3, places=6)

    def cells(self):
        cells = {}
        for root in spec.ROOTS:
            for scenario in spec.SCENARIOS:
                learned, forecast = control(), control(1.1)
                probes = []
                for index in spec.PROBES:
                    for alternative in spec.ALTERNATIVES:
                        row = control(.99 if alternative == "zero" else 1.01)
                        action = [0, 0] if alternative == "zero" else [-.3, .5]
                        row["trace"][index]["cost_before"] = learned["trace"][index]["cost_before"]
                        row["trace"][index]["action"] = action
                        probes.append({"alternative": alternative,
                            **spec.probe_result(ToyAgent(), learned, row, index, action)})
                cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                    "seed": root, "scenario": scenario,
                    "scene_seed": spec.source.source.source_spec.scene_seeds(spec.source.LOWER_ROOT[root], scenario, preflight=False)[0],
                    "software_qualified": True, "training_updates": 0, "native_steps": 9 * 61380,
                    "source_reproduced": list(spec.CONTROLS),
                    "controls": {"learned": learned, "forecast": forecast, "constant_residual": control(1.05)},
                    "probes": probes, "credit_diagnostics": spec.paired_credit(learned["trace"], forecast["trace"]),
                    "state_min": [0] * 34, "state_max": [1] * 34, "action_abs_gt_0_95_fraction": [0, 0]}
        return cells

    def test_analysis_requires_matched_interventions_and_complete_budget(self):
        cells = self.cells()
        result = summarize(cells)
        self.assertEqual(result["native_steps"], 90 * 61380)
        self.assertEqual(result["training_updates"], 0)
        self.assertEqual(len(result["panels"]), 10)
        self.assertEqual(result["panels"][0]["critic_deployment_rank_matches"], 3)
        for failure in ("prefix", "action", "duplicate", "budget", "paired_credit", "exogenous"):
            broken = copy.deepcopy(cells)
            c = broken[397, "low_noise"]
            if failure == "prefix": c["probes"][0]["prefix_matched"] = False
            elif failure == "action": c["probes"][0]["intervention_action"] = [1, 1]
            elif failure == "duplicate": c["probes"][0] = c["probes"][1]
            elif failure == "budget": c["training_updates"] = 1
            elif failure == "paired_credit": c["credit_diagnostics"]["paired_reward_sum"] += 1
            else: c["probes"][0]["passengers_generated"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken)

    def test_scheduler_keeps_code_only_unpinned_cpu_placement(self):
        task = task_specification("test_plan_credit", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual([p.rsplit("/", 1)[-1] for p in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])
        self.assertIn("--scenario low_noise", task["cmd"])


if __name__ == "__main__":
    unittest.main()
