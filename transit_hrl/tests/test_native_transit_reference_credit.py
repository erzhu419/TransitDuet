import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core.prefix_cost_credit import PrefixCostCredit
from freq_hrl.core.reference_credit import paired_prefix_credit
from scripts import run_native_transit_reference_credit_stage159 as spec
from scripts.analyze_native_transit_reference_credit_stage159 import summarize, validate_training
from scripts.submit_native_transit_reference_credit_stage159_scheduleurm import task_specification
from tests import test_native_transit_residual_plan as residual_tests


def ledger(costs):
    result = PrefixCostCredit()
    for i, cost in enumerate(costs[:-1]):
        result.begin([i], [i / 10, 0], cost, i * 1000)
    result.finish(costs[-1], 5400)
    return result


class ReferenceCreditTest(unittest.TestCase):
    def test_reference_changes_only_rewards_and_keeps_physical_terminal_objective(self):
        factual, reference = ledger([5, 8, 7, 1]), ledger([5, 8.1, 7.1, 1.2])
        original = [t["reward"] for t in factual.transitions]
        transitions, summary = paired_prefix_credit(factual, reference)
        self.assertAlmostEqual(summary["terminal_advantage"], 20)
        self.assertAlmostEqual(summary["replay_reward_sum"], 20)
        self.assertLess(summary["paired_reward_std"], summary["raw_reward_std"])
        self.assertEqual(original, [t["reward"] for t in factual.transitions])
        for a, b in zip(transitions, factual.transitions):
            for key in ("state", "action", "next_state"):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ("duration_s", "done", "cost_before", "cost_after"):
                self.assertEqual(a[key], b[key])

    def test_reference_rejects_unfinished_or_mismatched_clocks(self):
        a, b = ledger([5, 6, 2]), ledger([5, 6, 2])
        b.transitions[0]["duration_s"] += 1
        with self.assertRaises(ValueError): paired_prefix_credit(a, b)
        b = ledger([5, 6, 2])
        b.pending = {"time": 0}
        with self.assertRaises(ValueError): paired_prefix_credit(a, b)

    def fake_episode(self, root, scenario, scene, raw, checkpoint, action_fn, **kwargs):
        credit, decisions, cost = PrefixCostCredit(), [], 5.
        for i in range(4):
            state = np.full(34, i / 4, dtype=np.float32)
            action = action_fn(state)
            credit.begin(state, action, cost, i * 1000)
            decisions.append({"state": state, "action": action})
            cost -= (.3 if i % 2 == 0 else .7) - .02 * float(action.sum())
        summary = credit.finish(cost, 5400)
        row = {k: 1. for k in spec.source.source.source_spec.authority.routing.METRICS}
        row.update(service_cost_restricted=round(cost, 6), ep=300, N_fleet=12,
            simulation_end_time_s=5400, done_reason="evaluation_horizon", passengers_generated=100,
            passengers_unserved=0, trips_completed=24, ep_steps=100)
        return row, SimpleNamespace(credit=credit, decisions=decisions), summary, {"endpoint_error_max_s": 0}

    def test_extra_reference_rollouts_preserve_raw_sac_and_paired_sac_updates(self):
        with patch.object(spec.source, "episode", side_effect=self.fake_episode):
            original_agent, original = spec.source.train(397, Path("unused"), Path("unused"), preflight=True)
            raw_agent, raw = spec.train(397, Path("unused"), Path("unused"), credit_mode="raw", preflight=True)
            paired_agent, paired = spec.train(397, Path("unused"), Path("unused"), credit_mode="paired", preflight=True)
        spec.check_raw_reproduction(raw, original)
        for key, value in original_agent.state_dict().items():
            torch.testing.assert_close(value, raw_agent.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(paired_agent.update_step, 2)
        self.assertGreater(paired["actor_change_max_abs"], 0)
        validate_training(paired, 397, mode="paired", short=True)
        self.assertTrue(all(p["paired_reward_std"] < p["raw_reward_std"] for p in paired["credit_pairs"]))

    def cells(self):
        raw_cells = residual_tests.NativeResidualPlanTest().cells()
        cells = {}
        for root, raw in raw_cells.items():
            for r in raw["evaluation"]:
                r.update(passengers_unserved=0, trips_completed=262, ep_steps=100)
            short = copy.deepcopy(raw["preflight_learning"])
            short.update(transitions=8, training_scene_seeds=[600000000 + root * 1000 + ep for ep in range(2)],
                         final_actor_training_state_std=[.1, .1], training_action_std=[.1, .1])
            short["training_curve"][0].update({**copy.deepcopy(raw["evaluation"][0]), "episode": 0})
            short["training_curve"][0]["simulation_end_time_s"] = 5400
            short["training_curve"][0]["credit"].update(duration_s=5400, decisions=4)
            raw["preflight_learning"] = copy.deepcopy(short)

            def training(base, mode, count, decisions, clock):
                run = copy.deepcopy(base)
                run.update(credit_mode=mode, reference_episodes=count, transitions=count * decisions,
                    credit_pairs=[{"episode": ep, "scenario": spec.source.contract()["scenarios"][ep % 5],
                        "scene_seed": (600000000 if count == 2 else 700000000) + root * 1000 + ep,
                        "decisions": decisions, "duration_s": clock, "terminal_transitions": 1,
                        "reference_final_cost": 2, "controlled_final_cost": 1,
                        "terminal_advantage": 100, "paired_reward_sum": 100, "replay_reward_sum": 100,
                        "raw_reward_std": 100, "reference_reward_std": 100, "paired_reward_std": 1} for ep in range(count)])
                return run

            full = training(raw["training"], "paired", 120, 44, 61380)
            full["training_curve"][0]["episode"] = 0
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": root,
                "software_qualified": True, "native_training_updates": 0, "native_steps": 320 * 61380,
                "worker_preflight_native_steps": 61380 + 8 * 5400, "forecast_source_reproduced": True,
                "raw_short_reproduced": True, "raw_short": training(short, "raw", 2, 4, 5400),
                "paired_short": training(short, "paired", 2, 4, 5400), "training": full,
                "evaluation": copy.deepcopy(raw["evaluation"])}
        return cells, raw_cells

    def test_analysis_requires_reference_budget_and_replay_credit(self):
        cells, raw = self.cells()
        result = summarize(cells, raw)
        self.assertEqual(result["native_steps"], 640 * 61380)
        self.assertEqual(result["paired_minus_registered_raw"][0]["service_cost_restricted"], 0)
        for failure in ("reference", "credit", "budget", "raw_reproduction", "duplicate"):
            broken = copy.deepcopy(cells)
            c = broken[397]
            if failure == "reference": c["training"]["reference_episodes"] -= 1
            elif failure == "credit": c["training"]["credit_pairs"][0]["paired_reward_sum"] += 1
            elif failure == "budget": c["native_steps"] -= 61380
            elif failure == "raw_reproduction": c["raw_short"]["constant_action"] = [0, 0]
            else: c["evaluation"].append(copy.deepcopy(c["evaluation"][0]))
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, raw)

    def test_scheduler_is_unpinned_code_only(self):
        task = task_specification("test_reference", 397)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])


if __name__ == "__main__":
    unittest.main()
