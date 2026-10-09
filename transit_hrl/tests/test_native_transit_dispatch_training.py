import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import run_native_transit_dispatch_train_stage151 as spec
from scripts.analyze_native_transit_dispatch_train_stage151 import summarize
from scripts.submit_native_transit_dispatch_train_stage151_scheduleurm import task_specification


class NativeDispatchTrainingTest(unittest.TestCase):
    def test_factorial_changes_only_coupling_and_established_credit(self):
        base = {"frequency": {}, "env": {}, "coupling": {}, "upper": {}, "lower": {}}
        before = copy.deepcopy(base)
        configs = {method: spec.configure(base, method, 241, preflight=False) for method in spec.METHODS}
        self.assertEqual(base, before)
        for credit in ("", "_service_credit"):
            hiro, dispatch = configs["hiro" + credit], configs["dispatch" + credit]
            expected = copy.deepcopy(hiro)
            expected["coupling"]["coupling_mode"] = "channels"
            self.assertEqual(dispatch, expected)
        for cfg in configs.values():
            self.assertEqual(cfg["lower"]["state_encoder"]["input_schema"], "explicit_target_v2")
            self.assertEqual(cfg["frequency"]["routing"], "correct")
            self.assertFalse(cfg["env"]["allow_early_finish"])
        self.assertEqual(configs["dispatch_service_credit"]["upper"]["interval_credit"]["weights"]["headway"], 0)

    def test_worker_preflight_and_full_budgets_are_separate(self):
        self.assertEqual(spec.expected_updates(True), {"upper": 2, "lower": 4})
        self.assertEqual(spec.expected_updates(False), {"upper": 2700, "lower": 9000})
        self.assertEqual(spec.contract(True)["training_clock_s"], 5400)
        self.assertEqual(spec.contract(False)["training_clock_s"], 61380)
        self.assertEqual(spec.contract(False)["evaluation_episodes_per_scenario"], 4)
        all_scenes = [scene for root in spec.ROOTS for scenario in spec.contract(False)["scenarios"]
                      for scene in spec.scene_seeds(root, scenario, preflight=False)]
        self.assertEqual(len(all_scenes), len(set(all_scenes)))
        self.assertFalse(set(spec.ROOTS) & set(spec.authority.ROOTS))

    def test_worker_qualifies_before_full_training_and_stops_on_failure(self):
        argv = ["run", "--method", "dispatch", "--seed", "241", "--output", "/unused/result.json"]
        with patch.object(spec.sys, "argv", argv), patch.object(spec, "run_cell", side_effect=RuntimeError("qualification failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                spec.main()
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.kwargs["preflight"])
        with patch.object(spec.sys, "argv", argv), patch.object(spec, "write_json") as write, patch.object(
                spec, "run_cell", side_effect=[{"native_steps": 27000}, {}]) as run:
            spec.main()
            self.assertEqual([call.kwargs["preflight"] for call in run.call_args_list], [True, False])
            self.assertEqual([call.args[2] for call in run.call_args_list],
                             [Path("/unused/preflight.json"), Path("/unused/result.json")])
            self.assertTrue(write.call_args.args[1]["worker_preflight_passed"])
            self.assertEqual(write.call_args.args[1]["worker_preflight_native_steps"], 27000)

    def cells(self):
        cells = {}
        contract = spec.contract(False)
        for index, method in enumerate(spec.METHODS):
            for root in spec.ROOTS:
                rows = [{"condition": condition, "scenario": scenario, "scene_seed": scene,
                    "passengers_generated": 100, "N_fleet": 12, "simulation_end_time_s": 61380,
                    "dispatch": {"advance_count": 1, "delay_count": 2},
                    "command_abs_mean_s": 1, "subsecond_command_fraction": .5,
                    **{metric: index + (2 if condition == "neutral_upper" else 0)
                       for metric in (*spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean")}}
                    for scenario in contract["scenarios"] for scene in spec.scene_seeds(root, scenario, preflight=False)
                    for condition in spec.CONDITIONS]
                cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                    "method": method, "seed": root, "software_qualified": True,
                    "worker_preflight_passed": True, "worker_preflight_native_steps": 5 * 5400,
                    "updates": spec.expected_updates(False), "actor_dims": contract["actor_dims"],
                    "actor_change_max_abs": {"upper": .1, "lower": .1}, "parameter_counts": {"upper": 1, "lower": 1},
                    "training_demand_counts": [100] * 300, "training_fleets": [12] * 300,
                    "native_steps": (300 + len(rows)) * 61380, "evaluation": rows}
        return cells

    def test_matched_factor_and_upper_effects_use_root_not_episode_as_unit(self):
        result = summarize(self.cells())
        self.assertEqual(result["native_steps"], 16 * 360 * 61380)
        self.assertEqual(result["worker_preflight_native_steps"], 16 * 5 * 5400)
        self.assertIn("descriptive", result["stage"])
        self.assertEqual(len(result["regime_root_means"]), 16 * 3 * 5)
        for row in result["root_contrasts"]["dispatch_minus_hiro"]:
            self.assertEqual(row["service_cost_restricted"], 2)
        for row in result["credit_by_action_mode_interaction"]:
            self.assertEqual(row["service_cost_restricted"], 0)
        for method in spec.METHODS:
            self.assertEqual(len(result["learned_minus_neutral_upper"][method]), 4)
            self.assertEqual(result["learned_minus_neutral_upper"][method][0]["service_cost_restricted"], -2)

    def test_invalid_learning_pairing_or_scene_completeness_stops_merge(self):
        for failure in ("missing_cell", "missing_scene", "duplicate", "demand", "fleet", "updates",
                        "preflight", "actor", "clock", "nonfinite", "parameter_count"):
            with self.subTest(failure=failure):
                cells = self.cells()
                cell = cells["dispatch", spec.ROOTS[0]]
                if failure == "missing_cell":
                    cells.pop(("dispatch", spec.ROOTS[0]))
                elif failure == "missing_scene":
                    cell["evaluation"].pop()
                elif failure == "duplicate":
                    cell["evaluation"].append(copy.deepcopy(cell["evaluation"][0]))
                elif failure == "demand":
                    cell["training_demand_counts"][0] += 1
                elif failure == "fleet":
                    cell["training_fleets"][0] += 1
                elif failure == "updates":
                    cell["updates"]["upper"] = 0
                elif failure == "preflight":
                    cell["worker_preflight_passed"] = False
                elif failure == "actor":
                    cell["actor_change_max_abs"]["upper"] = 0
                elif failure == "clock":
                    cell["evaluation"][0]["simulation_end_time_s"] -= 1
                elif failure == "nonfinite":
                    cell["evaluation"][0]["service_cost_restricted"] = float("nan")
                else:
                    cell["parameter_counts"]["upper"] += 1
                with self.assertRaises(ValueError):
                    summarize(cells)

    def test_scheduler_stages_code_only_without_node_binding(self):
        for method in spec.METHODS:
            task = task_specification("test_dispatch_training", method, 241)
            self.assertEqual([path.split("/")[-1] for path in task["stage_input_paths"]],
                             ["scripts", "freq_hrl", "native_freqduet"])
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("run_native_transit_dispatch_train_stage151.py", task["cmd"])
            self.assertIn(f"--method {method}", task["cmd"])


if __name__ == "__main__":
    unittest.main()
