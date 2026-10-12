import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from native_freqduet.env.scenario import ScenarioTape
from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit, NativeTimetableContextAudit
from scripts import run_native_transit_continuation_noise_stage191 as spec
from scripts.analyze_native_transit_continuation_noise_stage191 import signal, state_panel, summarize
from scripts.submit_native_transit_continuation_noise_stage191_scheduleurm import task_specification, report_task_specification
from scripts import submit_native_transit_continuation_noise_stage191_scheduleurm as submit


def fixture(key):
    root, scenario, probe, draw = key
    def labels(cost):
        return [{"candidate": i, "actual_cost_delta": cost if i == 1 else 0.,
            "fleet_cost_delta": 0., "nonfleet_cost_delta": cost if i == 1 else 0.,
            "wait_delta_min": -.1 if i == 1 else 0., "episode_reward_delta": -100.*cost if i == 1 else 0.,
            "physical_return_delta": -100.*cost if i == 1 else 0., "future_fleet_cost": 0.} for i in range(6)]
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "root": root, "scenario": scenario, "scene_seed": spec.source.scene_seed(root, scenario, 0),
        "probe_index": probe, "draw": draw, "continuation_seed": spec.continuation_seed(*key),
        "original_choice": 0, "existing_labels": labels(-.4), "labels": labels(-.4 if draw % 2 == 0 else .4),
        "frozen_choices": {f"{stage}:{mode}:{split}": 1 for stage in ("stage188", "stage190")
            for mode in spec.bank_source.MODES for split in spec.bank_source.SPLITS},
        "original_source_reproductions": 1, "exact_resampled_identities": 1, "prefix_state_rng_matches": 7,
        "frozen_weight_qualified_episodes": 8, "window_ledger_qualified_episodes": 8, "continuation_qualified": True,
        "native_clock_ticks": 8*61380, "training_updates": 0}


class ContinuationNoiseTest(unittest.TestCase):
    def test_resampling_occurs_after_snapshot_only_at_registered_probe(self):
        plan = object.__new__(NativeContinuationValueAudit)
        old = ScenarioTape(100)
        old.poisson_stream(10., "arrivals", 1)
        profile = {6: 1.5}
        plan.env = SimpleNamespace(scenario_tape=old, current_time=1000., _demand_multipliers=profile, _peak_shift=1)
        plan.probe_index, plan.continuation_seed, plan.decisions = 8, 191, [None]*7
        def select(candidates):
            self.assertIs(plan.env.scenario_tape, old)
            plan.probe = {"index": len(plan.decisions)}
            return 3
        with patch.object(NativeTimetableContextAudit, "select_candidate", side_effect=select):
            self.assertEqual(plan.select_candidate([]), 3)
            self.assertIs(plan.env.scenario_tape, old)
            plan.decisions.append(None)
            self.assertEqual(plan.select_candidate([]), 3)
        self.assertEqual(plan.env.scenario_tape.seed, 191)
        self.assertFalse(plan.env.scenario_tape._streams)
        self.assertEqual(plan.probe["continuation_start_time_s"], 1000.)
        self.assertEqual(old.seed, 100)
        self.assertTrue(old._streams)
        self.assertIs(plan.env._demand_multipliers, profile)
        self.assertEqual(plan.env._peak_shift, 1)
        a, b, c = ScenarioTape(191), ScenarioTape(191), ScenarioTape(192)
        draws = lambda tape: [tape.normal_stream(0., 1., "speed", 1) for _ in range(16)]
        xa, xb, xc = draws(a), draws(b), draws(c)
        self.assertEqual(xa, xb)
        self.assertNotEqual(xa, xc)

    def test_seed_and_budget_count_correlated_states_not_new_scenes(self):
        keys = spec.cells()
        self.assertEqual(len(keys), 240)
        self.assertEqual(len(spec.cells(preflight=True)), 10)
        self.assertEqual(len({spec.continuation_seed(*k) for k in keys}), 240)
        self.assertEqual(len({(r, s, p) for r, s, p, d in keys}), 30)
        self.assertEqual(len({(r, s) for r, s, p, d in keys}), 10)
        self.assertEqual(spec.contract()["budget"]["native_episodes"], 1920)

    def test_effect_power_subtracts_sampling_noise_without_clipping_negative_estimate(self):
        constant = np.tile(np.arange(6), (8, 1))
        s = signal(constant)
        a, b = np.triu_indices(6, 1)
        self.assertEqual(s["debiased_pairwise_effect_power"], float(np.mean((constant[0, a]-constant[0, b])**2)))
        self.assertEqual(s["mean_pairwise_gap_variance"], 0.)
        noise = np.zeros((8, 6))
        noise[:, 1] = [1., -1.]*4
        s = signal(noise)
        self.assertEqual(s["observed_pairwise_squared_mean"], 0.)
        self.assertLess(s["debiased_pairwise_effect_power"], 0.)
        self.assertAlmostEqual(s["debiased_pairwise_effect_power"], -s["mean_label_pairwise_noise_variance"])

    def test_cross_half_oracle_exposes_optimism_and_frozen_choices_are_not_reselected(self):
        state = state_panel([fixture((397, spec.SCENARIOS[0], 22, d)) for d in spec.DRAWS])
        self.assertEqual(state["half_choices"], [1, 0])
        self.assertFalse(state["half_choice_agreement"])
        self.assertAlmostEqual(state["same_half_selected_cost"], -.2)
        self.assertAlmostEqual(state["cross_half_selected_cost"], .2)
        self.assertEqual(state["full_mean_oracle_choice"], 0)
        self.assertAlmostEqual(state["existing_label_oracle_old_cost"], -.4)
        self.assertAlmostEqual(state["existing_label_oracle_new_mean_cost"], 0.)
        self.assertTrue(all(p["selected"] == 1 for p in state["frozen_choice_outcomes"].values()))

    def test_qualification_rejects_changed_source_future_seed_identity_and_draw_choice(self):
        cells = [fixture((397, spec.SCENARIOS[0], 22, d)) for d in spec.DRAWS]
        state_panel(cells)
        for fault in ("seed", "scene", "identity", "sealed", "missing", "source", "budget"):
            broken = copy.deepcopy(cells)
            if fault == "seed": broken[0]["continuation_seed"] += 1
            elif fault == "scene": broken[0]["scene_seed"] += 1
            elif fault == "identity": broken[0]["labels"][0]["wait_delta_min"] = 1.
            elif fault == "sealed": broken[0]["frozen_choices"][next(iter(broken[0]["frozen_choices"]))] = 0
            elif fault == "source": broken[0]["existing_labels"][1]["episode_reward_delta"] += 1.
            elif fault == "budget": broken[0]["native_clock_ticks"] += 1
            else: broken.pop()
            with self.subTest(fault=fault), self.assertRaises(ValueError): state_panel(broken)

    def test_aggregate_does_not_promote_eight_draws_into_independent_scenes_or_claims(self):
        cells = {k: fixture(k) for k in spec.cells()}
        result = summarize(cells)
        self.assertEqual((result["overall"]["states"], result["overall"]["prefix_scene_groups"]), (30, 10))
        self.assertEqual((result["native_episodes"], result["training_updates"]), (1920, 0))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells)

    def test_scheduler_unbound_receipts_exclude_server_only_draw_labels_and_rollouts(self):
        tasks = [task_specification("unit_noise", *k) for k in spec.cells()]
        self.assertEqual(len({t["signature"] for t in tasks}), 240)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 240)
        for task in [*tasks, report_task_specification("unit_noise")]:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertNotIn("bank.json", task["cmd"])
        self.assertIn("/report", report_task_specification("unit_noise")["result_dir"])

    def test_remaining_uses_qualified_worker_exits_without_downloading_receipts(self):
        tasks = [task_specification("unit_noise", *k) for k in spec.cells(preflight=True)]
        for task in tasks:
            task.update(status="done", exit_code=0)
        with patch.object(submit, "inventory", return_value=tasks), \
                patch.object(Path, "read_text", side_effect=AssertionError("No receipt download needed")), \
                patch.object(submit, "execute_bulk") as execute, \
                patch.object(sys, "argv", ["submit", "--run-name", "unit_noise", "--phase", "remaining"]):
            submit.main()
        self.assertEqual(len(execute.call_args.args[0]), 230)
        tasks[0]["exit_code"] = 1
        with patch.object(submit, "inventory", return_value=tasks), \
                patch.object(submit, "execute_bulk") as execute, \
                patch.object(sys, "argv", ["submit", "--run-name", "unit_noise", "--phase", "remaining"]):
            with self.assertRaises(SystemExit): submit.main()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
