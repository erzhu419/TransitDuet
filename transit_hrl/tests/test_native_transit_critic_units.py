import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from native_freqduet.upper.resac_upper import (
    EnsembleQNetwork, UnitActionEnsembleQNetwork, RESACUpperTrainer,
)
from freq_hrl.domains.transit.native_value_diagnostics import critic_input_scale
from scripts import run_native_transit_critic_units_stage152 as spec
from scripts.analyze_native_transit_critic_units_stage152 import summarize
from scripts.submit_native_transit_critic_units_stage152_scheduleurm import task_specification


class NativeCriticUnitsTest(unittest.TestCase):
    def trainer(self, units):
        torch.manual_seed(9)
        return RESACUpperTrainer(state_dim=16, action_dim=1,
            action_low=[-120], action_high=[120], critic_action_units=units)

    def test_unit_critic_has_exact_coordinates_and_chain_rule(self):
        torch.manual_seed(7)
        unit = UnitActionEnsembleQNetwork(16, 1, [-120], [120])
        torch.manual_seed(7)
        native = EnsembleQNetwork(16, 1)
        self.assertEqual(sum(p.numel() for p in unit.parameters()), sum(p.numel() for p in native.parameters()))
        state = torch.ones(3, 16)
        physical = torch.tensor([[-120.], [0.], [120.]], requires_grad=True)
        coordinates = torch.tensor([[-1.], [0.], [1.]], requires_grad=True)
        torch.testing.assert_close(unit.action_coordinates(physical), coordinates)
        actual, expected = unit(state, physical), native(state, coordinates)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        physical_gradient = torch.autograd.grad(actual.sum(), physical)[0]
        unit_gradient = torch.autograd.grad(expected.sum(), coordinates)[0]
        torch.testing.assert_close(physical_gradient, unit_gradient / 120)
        shifted = UnitActionEnsembleQNetwork(16, 1, [0], [60])
        torch.testing.assert_close(shifted.action_coordinates(torch.tensor([[0.], [30.], [60.]])), coordinates)

    def test_both_coordinate_modes_preserve_actor_initialization_and_physical_output(self):
        native, unit = self.trainer("seconds"), self.trainer("unit")
        for name, value in native.policy_net.state_dict().items():
            torch.testing.assert_close(value, unit.policy_net.state_dict()[name], rtol=0, atol=0)
        for name, value in native.q_net.named_parameters():
            torch.testing.assert_close(value, dict(unit.q_net.named_parameters())[name], rtol=0, atol=0)
        state = torch.ones(16)
        np.testing.assert_array_equal(native.policy_net.get_action(state, deterministic=True),
                                      unit.policy_net.get_action(state, deterministic=True))
        self.assertEqual(native.policy_net.action_low.item(), -120)
        self.assertEqual(unit.policy_net.action_high.item(), 120)

    def test_first_layer_action_contribution_changes_by_exact_range_not_state(self):
        native, unit = self.trainer("seconds"), self.trainer("unit")
        states = np.ones((4, 16), dtype=np.float32)
        a, b = critic_input_scale(native, states), critic_input_scale(unit, states)
        self.assertEqual(a["state_contribution_abs_mean"], b["state_contribution_abs_mean"])
        self.assertAlmostEqual(a["action_contribution_abs_mean"] / b["action_contribution_abs_mean"], 120, places=4)

    def test_updates_keep_replay_seconds_and_report_actual_target_clipping(self):
        for units in ("seconds", "unit"):
            trainer = self.trainer(units)
            state = np.ones(16, dtype=np.float32)
            for _ in range(8):
                trainer.replay_buffer.push(state, [60.], 100., state, True)
            self.assertEqual(trainer.replay_buffer.buffer[0][1].item(), 60)
            before = {name: value.clone() for name, value in trainer.policy_net.state_dict().items()}
            metrics = trainer.update(batch_size=8)
            self.assertEqual(metrics["upper_target_clip_fraction"], 1)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(any(not torch.equal(value, before[name]) for name, value in trainer.policy_net.state_dict().items()))
            if units == "unit":
                self.assertTrue(torch.equal(trainer.q_net.action_radius, trainer.target_q_net.action_radius))

    def test_checkpoint_roundtrip_keeps_units_and_rejects_other_coordinate_system(self):
        with tempfile.TemporaryDirectory() as directory:
            for units in ("seconds", "unit"):
                trainer = self.trainer(units)
                path = Path(directory) / f"{units}.pt"
                trainer.save(path)
                restored = self.trainer(units)
                restored.load(path)
                for name, value in trainer.q_net.state_dict().items():
                    torch.testing.assert_close(value, restored.q_net.state_dict()[name], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "action units"):
                    self.trainer("unit" if units == "seconds" else "seconds").load(path)
            legacy = torch.load(Path(directory) / "seconds.pt", weights_only=True)
            legacy.pop("critic_action_units")
            path = Path(directory) / "stage151.pt"
            torch.save(legacy, path)
            self.trainer("seconds").load(path)

    def test_factorial_changes_only_units_within_each_credit_mode(self):
        base = {"frequency": {}, "env": {}, "coupling": {}, "upper": {}, "lower": {}}
        before = copy.deepcopy(base)
        for method in ("dispatch", "dispatch_service_credit"):
            physical = spec.configure(base, method, spec.ROOTS[0], preflight=False)
            unit = spec.configure(base, method + "_unit", spec.ROOTS[0], preflight=False)
            expected = copy.deepcopy(physical)
            expected["upper"]["critic_action_units"] = "unit"
            self.assertEqual(unit, expected)
            self.assertEqual(unit["coupling"]["coupling_mode"], "channels")
        self.assertEqual(base, before)
        self.assertFalse(set(spec.ROOTS) & set(spec.training.ROOTS))
        self.assertEqual(spec.expected_updates(False), {"upper": 2700, "lower": 9000})

    def cells(self):
        contract, cells = spec.contract(False), {}
        metrics = (*spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean",
                   "command_abs_mean_s", "subsecond_command_fraction")
        for index, method in enumerate(spec.METHODS):
            units = contract["critic_action_units"][method]
            for root in spec.ROOTS:
                rows = [{"condition": condition, "scenario": scenario, "scene_seed": scene,
                    "N_fleet": 12, "passengers_generated": 100, "simulation_end_time_s": 61380,
                    "dispatch": {"advance_count": 1, "delay_count": 2},
                    "critic": {"input_scale": {"critic_action_units": units,
                        "state_contribution_abs_mean": .1, "action_contribution_abs_mean": .1}},
                    **{key: index + (2 if condition == "neutral_upper" else 0) for key in metrics}}
                    for scenario in contract["scenarios"] for scene in spec.scene_seeds(root, scenario, preflight=False)
                    for condition in spec.CONDITIONS]
                cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                    "method": method, "seed": root, "software_qualified": True,
                    "worker_preflight_passed": True, "worker_preflight_native_steps": 27000,
                    "updates": spec.expected_updates(False), "actor_dims": contract["actor_dims"],
                    "actor_change_max_abs": {"upper": .1, "lower": .1}, "parameter_counts": {"upper": 1, "lower": 1},
                    "training_demand_counts": [100] * 300, "training_fleets": [12] * 300,
                    "critic_action_units": units, "training_curve": [{"upper_learning": {
                        "upper_target_clip_fraction": 0, "upper_q_mse": 1}}],
                    "upper_learning_updates": 2700,
                    "upper_learning_mean": {"upper_target_clip_fraction": 0, "upper_q_mse": 1},
                    "native_steps": 360 * 61380, "evaluation": rows}
        return cells

    def test_analysis_separates_training_contrast_from_own_upper_control(self):
        result = summarize(self.cells())
        self.assertEqual(result["native_steps"], 176774400)
        self.assertEqual(result["worker_preflight_native_steps"], 216000)
        self.assertEqual(len(result["critic_diagnostics"]), 8)
        for base in ("dispatch", "dispatch_service_credit"):
            self.assertEqual(result["unit_minus_seconds"][base][0]["service_cost_restricted"], 1)
        for method in spec.METHODS:
            self.assertEqual(result["learned_minus_neutral_upper"][method][0]["service_cost_restricted"], -2)

    def test_analysis_rejects_missing_pair_wrong_units_or_missing_diagnostics(self):
        for failure in ("missing", "units", "diagnostics", "learning", "clip", "demand", "successful_updates"):
            with self.subTest(failure=failure):
                cells = self.cells()
                cell = cells["dispatch_unit", spec.ROOTS[0]]
                if failure == "missing":
                    cells.pop(("dispatch", spec.ROOTS[0]))
                elif failure == "units":
                    cell["critic_action_units"] = "seconds"
                elif failure == "diagnostics":
                    cell["evaluation"][0]["critic"]["input_scale"]["state_contribution_abs_mean"] = float("nan")
                elif failure == "learning":
                    cell["training_curve"][-1]["upper_learning"] = {}
                elif failure == "clip":
                    cell["training_curve"][-1]["upper_learning"]["upper_target_clip_fraction"] = 2
                elif failure == "successful_updates":
                    cell["upper_learning_updates"] = 2699
                else:
                    cell["training_demand_counts"][0] += 1
                with self.assertRaises(ValueError):
                    summarize(cells)

    def test_worker_does_not_start_full_training_if_qualification_fails(self):
        argv = ["run", "--method", "dispatch_unit", "--seed", "293", "--output", "/unused/result.json"]
        with patch.object(spec.sys, "argv", argv), patch.object(spec.training, "run_cell",
                side_effect=RuntimeError("qualification failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                spec.main()
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.kwargs["preflight"])
            self.assertEqual(run.call_args.kwargs["experiment"], spec)

    def test_task_stages_code_only_with_all_compute_nodes_eligible(self):
        for method in spec.METHODS:
            task = task_specification("test_critic_units", method, 293)
            self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("run_native_transit_critic_units_stage152.py", task["cmd"])
            self.assertEqual(task["cpu"], 1)


if __name__ == "__main__":
    unittest.main()
