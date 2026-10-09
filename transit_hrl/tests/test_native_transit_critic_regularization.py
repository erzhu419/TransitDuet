import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from native_freqduet.upper.resac_upper import EnsembleQNetwork, UnitActionEnsembleQNetwork, RESACUpperTrainer
from freq_hrl.domains.transit.native_value_diagnostics import critic_action_curve
from scripts import run_native_transit_critic_regularization_stage153 as spec
from scripts.analyze_native_transit_critic_regularization_stage153 import summarize
from scripts.submit_native_transit_critic_regularization_stage153_scheduleurm import task_specification


class StateDependentQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state, action):
        return (state[:, 0] * action[:, 0] + self.anchor)[None].expand(10, -1)


class NativeCriticRegularizationTest(unittest.TestCase):
    def test_physical_penalty_matches_equivalent_seconds_function_and_affine_bias(self):
        for low, high in ((-120., 120.), (0., 60.)):
            torch.manual_seed(4)
            unit = UnitActionEnsembleQNetwork(16, 1, [low], [high])
            torch.manual_seed(4)
            seconds = EnsembleQNetwork(16, 1)
            with torch.no_grad():
                unit.biases[0].fill_(.1)
                seconds.weights[0].copy_(unit.weights[0])
                seconds.weights[0][:, -1:] /= unit.action_radius[None, :, None]
                seconds.biases[0].copy_(unit.biases[0] - (
                    seconds.weights[0][:, -1:] * unit.action_center[None, :, None]
                ).sum(1, keepdim=True))
            states, actions = torch.randn(7, 16), torch.linspace(low, high, 7)[:, None]
            torch.testing.assert_close(unit(states, actions), seconds(states, actions))
            torch.testing.assert_close(unit.compute_l1_norm("physical_sum"), seconds.compute_l1_norm("sum"))
            self.assertTrue(torch.all(unit.compute_l1_norm("sum") != unit.compute_l1_norm("physical_sum")))

    def test_action_penalty_gradient_uses_physical_units_without_weakening_state_columns(self):
        q = UnitActionEnsembleQNetwork(16, 1, [-120], [120])
        gradient = torch.autograd.grad(q.compute_l1_norm("physical_sum").sum(), q.weights[0])[0]
        expected = q.weights[0].detach().sign()
        expected[:, -1:] /= 120
        torch.testing.assert_close(gradient, expected)
        native = EnsembleQNetwork(16, 1)
        torch.testing.assert_close(native.compute_l1_norm("physical_sum"), native.compute_l1_norm("sum"), rtol=0, atol=0)
        torch.testing.assert_close(q.compute_l1_norm("mean"), q.compute_l1_norm("sum") / 9537)

    def test_physical_regularized_upper_update_is_finite_and_keeps_physical_replay(self):
        trainer = RESACUpperTrainer(16, 1, action_low=[-120], action_high=[120],
            critic_action_units="unit", weight_reg_mode="physical_sum")
        for i in range(8):
            trainer.replay_buffer.push(np.ones(16), [float(i * 10)], -.3, np.ones(16), i == 7)
        metrics = trainer.update(8)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertEqual(trainer.replay_buffer.buffer[-1][1].item(), 70)
        self.assertGreater(metrics["upper_q_l1_penalty"], 0)

    def test_paired_q_diagnostic_does_not_hide_opposing_state_effects(self):
        trainer = SimpleNamespace(q_net=StateDependentQ(), beta=-2)
        states = [np.array([value, *([0.] * 15)]) for value in (-1, 1)]
        before = torch.get_rng_state().clone()
        curve = critic_action_curve(trainer, states, [-120, 0, 120])
        self.assertTrue(all(r["q_mean"] == 0 for r in curve.values()))
        self.assertEqual(curve["120"]["q_difference_to_zero_abs_mean"], 120)
        self.assertEqual(curve["0"]["q_difference_to_zero_abs_p95"], 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_factors_keep_same_credit_coefficient_and_all_other_configuration(self):
        base = {"frequency": {}, "env": {}, "coupling": {}, "upper": {"weight_reg": .01}, "lower": {}}
        before = copy.deepcopy(base)
        configs = {method: spec.configure(base, method, 313, preflight=False) for method in spec.METHODS}
        for method, cfg in configs.items():
            expected = copy.deepcopy(configs["seconds_sum"])
            units, mode = spec.COORDINATES[method]
            expected["upper"].update(critic_action_units=units, weight_reg_mode=mode)
            self.assertEqual(cfg, expected)
            self.assertEqual(cfg["upper"]["weight_reg"], .01)
            self.assertEqual(cfg["upper"]["interval_credit"]["weights"]["headway"], 0)
        self.assertEqual(base, before)
        self.assertFalse(set(spec.ROOTS) & set(spec.source.ROOTS))

    def cells(self):
        contract, cells = spec.contract(False), {}
        metrics = (*spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean",
                   "command_abs_mean_s", "subsecond_command_fraction")
        for index, method in enumerate(spec.METHODS):
            units, mode = spec.COORDINATES[method]
            for root in spec.ROOTS:
                rows = [{"condition": condition, "scenario": scenario, "scene_seed": scene,
                    "N_fleet": 12, "passengers_generated": 100, "simulation_end_time_s": 61380,
                    "dispatch": {"advance_count": 1, "delay_count": 2},
                    "critic": {"input_scale": {"critic_action_units": units,
                        "state_contribution_abs_mean": .1, "action_contribution_abs_mean": .1},
                        "action_curve": {str(g): {"q_mean": 0, "q_difference_to_zero_abs_mean": abs(g)} for g in (-120, 0, 120)}},
                    **{key: index + (2 if condition == "neutral_upper" else 0) for key in metrics}}
                    for scenario in contract["scenarios"] for scene in spec.scene_seeds(root, scenario, preflight=False)
                    for condition in spec.CONDITIONS]
                cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                    "method": method, "seed": root, "software_qualified": True,
                    "worker_preflight_passed": True, "worker_preflight_native_steps": 27000,
                    "updates": spec.expected_updates(False), "actor_dims": contract["actor_dims"],
                    "actor_change_max_abs": {"upper": .1, "lower": .1}, "parameter_counts": {"upper": 1, "lower": 1},
                    "training_demand_counts": [100] * 300, "training_fleets": [12] * 300,
                    "critic_action_units": units, "weight_reg_mode": mode,
                    "training_curve": [{"upper_learning": {"upper_target_clip_fraction": 0, "upper_q_mse": 1}}],
                    "upper_learning_updates": 2700,
                    "upper_learning_mean": {"upper_target_clip_fraction": 0, "upper_q_mse": 1},
                    "native_steps": 360 * 61380, "evaluation": rows}
        return cells

    def test_analysis_keeps_mechanism_effect_distinct_from_own_upper_authority(self):
        result = summarize(self.cells())
        self.assertEqual(result["native_steps"], 176774400)
        self.assertEqual(result["worker_preflight_native_steps"], 216000)
        self.assertEqual(result["root_contrasts"]["physical_minus_unit"][0]["service_cost_restricted"], 1)
        self.assertEqual(result["root_contrasts"]["mean_minus_unit"][0]["service_cost_restricted"], 2)
        for method in spec.METHODS:
            self.assertEqual(result["learned_minus_neutral_upper"][method][0]["service_cost_restricted"], -2)
        self.assertEqual(result["critic_diagnostics"][0]["paired_q_difference_to_zero_abs_mean"]["120"], 120)

    def test_wrong_regularizer_failed_updates_or_missing_pair_stops_analysis(self):
        for failure in ("regularizer", "updates", "pair"):
            cells = self.cells()
            cell = cells["unit_physical", 313]
            if failure == "regularizer":
                cell["weight_reg_mode"] = "sum"
            elif failure == "updates":
                cell["upper_learning_updates"] = 2699
            else:
                cell["evaluation"].pop()
            with self.assertRaises(ValueError):
                summarize(cells)

    def test_qualification_failure_does_not_start_full_training(self):
        argv = ["run", "--method", "unit_physical", "--seed", "313", "--output", "/unused/result.json"]
        with patch.object(spec.sys, "argv", argv), patch.object(spec.source.training, "run_cell",
                side_effect=RuntimeError("qualification failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                spec.main()
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.kwargs["preflight"])
            self.assertEqual(run.call_args.kwargs["experiment"], spec)

    def test_scheduler_keeps_all_compute_nodes_eligible_and_stages_no_results(self):
        for method in spec.METHODS:
            task = task_specification("test_physical_regularizer", method, 313)
            self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["cpu"], 1)
            self.assertIn("run_native_transit_critic_regularization_stage153.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
