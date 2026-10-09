from types import SimpleNamespace
import unittest

import numpy as np
import torch

from freq_hrl.domains.transit.native_value_diagnostics import critic_action_curve, credit_ledger
from scripts import run_native_transit_frontier_stage149 as spec
from scripts.analyze_native_transit_frontier_stage149 import summarize
from scripts.submit_native_transit_frontier_stage149_scheduleurm import task_specification


class GoalQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state, action):
        values = -torch.abs(action[:, 0] - 15) + .1 * state[:, 0] + self.anchor
        return values.unsqueeze(0) + torch.arange(10).reshape(-1, 1)


class NativeFrontierTest(unittest.TestCase):
    def test_same_state_critic_query_preserves_rng_and_weights(self):
        trainer = SimpleNamespace(q_net=GoalQ(), beta=-2.0)
        rng = torch.get_rng_state().clone()
        weight = trainer.q_net.anchor.detach().clone()
        curve = critic_action_curve(trainer, [np.ones(16)] * 5, spec.GOALS)
        self.assertEqual(max(curve, key=lambda key: curve[key]["lcb_mean"]), "15")
        self.assertEqual(curve["15"]["lcb_argmax_state_fraction"], 1)
        self.assertAlmostEqual(sum(row["lcb_argmax_state_fraction"] for row in curve.values()), 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(weight, trainer.q_net.anchor))

    def test_ledger_exposes_long_final_interval_and_wait_share(self):
        rows = [{"duration_s": duration, "interval_wait_cost": wait,
                 "system_reward": -1, "gap_credit": 0, "interval_reward": -100 * wait}
                for duration, wait in ((180, .1), (180, .2), (14400, .7))]
        ledger = credit_ledger(rows)
        self.assertEqual(ledger["duration_median_s"], 180)
        self.assertEqual(ledger["last_duration_s"], 14400)
        self.assertAlmostEqual(ledger["last_wait_credit_fraction"], .7)
        self.assertEqual(ledger["interval_reward_sum"], -100)

    def fixtures(self, preflight=False):
        cells, sources = {}, {}
        contract = spec.contract(preflight)
        pairs = [(spec.METHODS[0], spec.ROOTS[0])] if preflight else [
            (method, root) for method in spec.METHODS for root in spec.ROOTS]
        for method, root in pairs:
            source_rows, new_rows = [], []
            for scenario in contract["scenarios"]:
                baseline = {key: 1 for key in (*spec.source_spec.routing.METRICS,
                    "passengers_unserved", "trips_completed", "ep_steps", "lower_action_mean", "upper_delta_mean")}
                baseline.update(scenario=scenario, scene_seed=spec.source_spec.scene_seed(root, scenario),
                    simulation_end_time_s=61380, passengers_generated=100, N_fleet=12)
                for condition in ("baseline", *spec.CACHED_GOALS.values()):
                    row = dict(baseline, condition=condition)
                    if condition in ("upper_minus60", "upper_plus60"):
                        row["service_cost_restricted"] = 2
                    source_rows.append(row)
                for condition in contract["new_conditions"]:
                    row = dict(baseline, condition=condition)
                    if condition == 15:
                        row["service_cost_restricted"] = .8
                    if condition == "baseline":
                        row["critic_curve"] = {str(goal): {"lcb_mean": -abs(goal)} for goal in spec.GOALS}
                        row["credit_ledger"] = {"last_duration_s": 14400}
                    new_rows.append(row)
            cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                "method": method, "seed": root, "software_qualified": True, "baseline_reproduced": True,
                "training_updates": 0, "native_steps": len(new_rows) * 61380, "evaluation": new_rows}
            sources[method, root] = {"evaluation": source_rows}
        return cells, sources

    def test_cached_endpoints_and_oracle_are_diagnostic_not_policy_evidence(self):
        cells, sources = self.fixtures()
        result = summarize(cells, sources, preflight=False)
        self.assertEqual(result["native_steps"], 100 * 61380)
        self.assertEqual(result["training_updates"], 0)
        self.assertEqual(len(result["panels"]), 20)
        self.assertEqual(result["stage"], "post_result_opportunity_diagnosis")
        for panel in result["panels"]:
            self.assertEqual(panel["oracle_grid_goal_s"], 15)
            self.assertEqual(panel["mean_state_critic_lcb_goal_s"], 0)
            self.assertAlmostEqual(panel["oracle_cost_gain_against_zero"], .2)
            self.assertEqual(len(panel["goals"]), 7)
        for row in result["root_opportunity"]:
            self.assertEqual(row["best_fixed_grid_goal_s"], 15)

    def test_preflight_does_not_select_goals_or_claim_performance(self):
        cells, sources = self.fixtures(True)
        result = summarize(cells, sources, preflight=True)
        self.assertEqual(result["native_steps"], 61380)
        self.assertNotIn("panels", result)

    def test_wrong_source_baseline_missing_duplicate_or_changed_scene_stops_merge(self):
        for failure in ("baseline", "missing", "duplicate", "scene"):
            cells, sources = self.fixtures()
            rows = cells[spec.METHODS[0], spec.ROOTS[0]]["evaluation"]
            if failure == "baseline":
                rows[0]["ep_reward"] = 2
            elif failure == "missing":
                rows.pop()
            elif failure == "duplicate":
                rows.append(dict(rows[0]))
            else:
                rows[-1]["scene_seed"] += 1
            with self.assertRaises((ValueError, RuntimeError)):
                summarize(cells, sources, preflight=False)

    def test_scheduler_stages_code_only_without_node_binding(self):
        task = task_specification("test_frontier", spec.METHODS[0], spec.ROOTS[0], preflight=True)
        self.assertEqual([path.split("/")[-1] for path in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(len(task["allowed_nodes"]), 6)
        self.assertIn("run_native_transit_frontier_stage149.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
