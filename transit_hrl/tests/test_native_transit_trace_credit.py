import copy
import random
import subprocess
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core.retrace import SequenceReplayBuffer, retrace_targets
from native_freqduet.upper.resac_upper import RESACUpperTrainer
from scripts import run_native_transit_trace_credit_stage155 as spec
from scripts.analyze_native_transit_trace_credit_stage155 import summarize
from scripts.submit_native_transit_trace_credit_stage155_scheduleurm import task_specification


class NativeTraceCreditTest(unittest.TestCase):
    def target(self, reward, discount, *, ratio=None, valid=None, trace_lambda=1.0, next_value=None):
        reward = torch.tensor([reward], dtype=torch.float64)
        zeros = torch.zeros_like(reward)
        log_pi = zeros if ratio is None else torch.log(torch.tensor([ratio], dtype=torch.float64))
        return retrace_targets(reward, torch.tensor([discount], dtype=torch.float64),
            zeros if next_value is None else torch.tensor([next_value], dtype=torch.float64),
            zeros, log_pi, zeros, torch.ones_like(reward) if valid is None else torch.tensor([valid]), trace_lambda)[0]

    def test_delayed_reward_and_next_action_importance_ratios(self):
        self.assertAlmostEqual(self.target([0, 0, 10], [.9, .9, 0])[0, 0].item(), 8.1)
        # c_0 never weights its own update, while c_1 and c_2 do.
        value = self.target([0, 0, 10], [.9, .9, 0], ratio=[.001, .5, .5])[0, 0].item()
        self.assertAlmostEqual(value, 2.025)
        self.assertAlmostEqual(self.target([0, 0, 10], [.9, .9, 0], ratio=[10, 100, 100])[0, 0].item(), 8.1)

    def test_lambda_zero_terminal_and_truncated_soft_bootstrap(self):
        value = self.target([1, 2, 999], [.9, .9, 0], valid=[1, 1, 0], next_value=[3, 4, 100])
        self.assertAlmostEqual(value[0, 0].item(), 1 + .9 * 3 + .9 * (2 + .9 * 4))
        zero = self.target([1, 2, 999], [.9, .9, 0], trace_lambda=0, next_value=[3, 4, 100])
        self.assertAlmostEqual(zero[0, 0].item(), 3.7)
        terminal = self.target([1, 2, 999], [0, .9, 0], next_value=[3, 4, 100])
        self.assertEqual(terminal[0, 0].item(), 1)
        physical = self.target([0, 10], [.9 ** 2, 0])
        self.assertAlmostEqual(physical[0, 0].item(), 8.1)

    def test_sequences_stop_at_episode_end_or_replay_frontier(self):
        replay = SequenceReplayBuffer(6)
        for state, done in ((0, False), (1, True), (100, False), (101, True), (200, False)):
            replay.push([state], [60], state, [state + 1], done, behavior_log_prob=-1)
        with patch("freq_hrl.core.retrace.random.sample", return_value=[0, 1, 4]):
            batch = replay.sample_sequences(3, 8)
        np.testing.assert_array_equal(batch["valid"].sum(axis=1), [2, 1, 1])
        np.testing.assert_array_equal(batch["state"][0, :2, 0], [0, 1])
        self.assertTrue(np.all(batch["action"] == 60))
        with self.assertRaises(ValueError):
            replay.push([0], [0], 0, [1], False, behavior_log_prob=float("nan"))

    def test_reference_one_step_optimizer_is_bit_exact(self):
        reference = ModuleType("preserved_stage154_upper")
        source = subprocess.check_output(["git", "show", "fa3e94d9af:transit_hrl/native_freqduet/upper/resac_upper.py"],
                                         cwd=spec.ROOT, text=True)
        exec(compile(source, "preserved_stage154_upper", "exec"), reference.__dict__)
        stats, arrays = [], []
        for constructor in (reference.RESACUpperTrainer, RESACUpperTrainer):
            torch.manual_seed(31)
            random.seed(31)
            trainer = constructor(16, 1, action_low=[-120], action_high=[120],
                                  critic_action_units="unit", weight_reg_mode="physical_sum")
            for i in range(16):
                trainer.replay_buffer.push(np.full(16, i / 16), [float(i)], -.3,
                                           np.full(16, (i + 1) / 16), i == 15)
            stats.append(trainer.update(8))
            arrays.append({(name, key): value.detach().clone() for name in ("policy_net", "q_net", "target_q_net")
                           for key, value in getattr(trainer, name).state_dict().items()})
        self.assertEqual(stats[0], stats[1])
        for key in arrays[0]:
            torch.testing.assert_close(arrays[0][key], arrays[1][key], rtol=0, atol=0)

    def test_corrected_backup_updates_both_actor_and_critic_with_finite_diagnostics(self):
        torch.manual_seed(37)
        random.seed(37)
        trainer = RESACUpperTrainer(16, 1, action_low=[-120], action_high=[120],
            critic_action_units="unit", weight_reg_mode="physical_sum", backup_horizon=8)
        for i in range(16):
            state, next_state = np.full(16, i / 16, dtype=np.float32), np.full(16, (i + 1) / 16, dtype=np.float32)
            action = trainer.policy_net.get_action(state)
            log_mu = float(trainer.policy_net.log_prob(state, action))
            trainer.replay_buffer.push(state, action, -1 if i == 15 else -.01, next_state,
                                      i == 15, behavior_log_prob=log_mu)
        before = {name: [p.detach().clone() for p in getattr(trainer, name).parameters()]
                  for name in ("policy_net", "q_net")}
        metrics = trainer.update(8)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertGreater(metrics["upper_trace_mass_mean"], 1)
        self.assertGreater(metrics["upper_trace_correction_abs_mean"], 0)
        self.assertAlmostEqual(metrics["upper_trace_coefficient_mean"], .9, places=5)
        for name, values in before.items():
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(values, getattr(trainer, name).parameters())))

    def test_configuration_changes_only_backup_horizon(self):
        base = {"frequency": {}, "env": {}, "coupling": {}, "upper": {}, "lower": {}}
        a = spec.configure(base, "one_step", 347, preflight=False)
        b = spec.configure(base, "retrace8", 347, preflight=False)
        expected = copy.deepcopy(a)
        expected["upper"]["backup_horizon"] = 8
        self.assertEqual(expected, b)
        self.assertEqual(spec.expected_updates(False), {"upper": 2700, "lower": 9000})

    def cells(self):
        contract, cells = spec.contract(False), {}
        metrics = (*spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean",
                   "command_abs_mean_s", "subsecond_command_fraction")
        for index, method in enumerate(spec.METHODS):
            for root in spec.ROOTS:
                rows = [{"condition": condition, "scenario": scenario, "scene_seed": scene,
                    "N_fleet": 12, "passengers_generated": 100, "simulation_end_time_s": 61380,
                    "dispatch": {"advance_count": 0, "delay_count": 1},
                    "critic": {"input_scale": {"critic_action_units": "unit",
                        "state_contribution_abs_mean": .1, "action_contribution_abs_mean": .1}},
                    **{key: index + (2 if condition == "neutral_upper" else 3 if condition == "fixed7" else 0)
                       for key in metrics}}
                    for scenario in contract["scenarios"] for scene in spec.scene_seeds(root, scenario, preflight=False)
                    for condition in spec.CONDITIONS]
                learning = {"upper_target_clip_fraction": 0, "upper_q_mse": 1}
                if method == "retrace8":
                    learning.update(upper_trace_mass_mean=3, upper_trace_valid_steps_mean=7,
                                    upper_trace_coefficient_mean=.8, upper_trace_correction_abs_mean=.2)
                cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                    "method": method, "seed": root, "software_qualified": True,
                    "worker_preflight_passed": True, "worker_preflight_native_steps": 32400,
                    "updates": spec.expected_updates(False), "actor_dims": contract["actor_dims"],
                    "actor_change_max_abs": {"upper": .1, "lower": .1}, "parameter_counts": {"upper": 1, "lower": 2},
                    "training_demand_counts": [100] * 300, "training_fleets": [12] * 300,
                    "critic_action_units": "unit", "weight_reg_mode": "physical_sum", "backup_horizon": spec.HORIZONS[method],
                    "trace_lambda": .9, "training_curve": [{"upper_learning": dict(learning)}],
                    "upper_learning_updates": 2700, "upper_learning_mean": learning,
                    "native_steps": 380 * 61380, "evaluation": rows}
        return cells

    def test_analysis_separates_full_training_and_same_checkpoint_controls(self):
        result = summarize(self.cells())
        self.assertEqual(result["native_steps"], 93297600)
        self.assertEqual(result["worker_preflight_native_steps"], 129600)
        self.assertEqual(result["retrace_minus_one_step"][0]["service_cost_restricted"], 1)
        self.assertEqual(result["learned_minus_fixed7"]["retrace8"][0]["service_cost_restricted"], -3)
        for failure in ("backup_horizon", "worker_preflight_native_steps", "correction"):
            cells = self.cells()
            cell = cells["retrace8", spec.ROOTS[0]]
            if failure == "correction":
                cell["upper_learning_mean"]["upper_trace_correction_abs_mean"] = 0
            else:
                cell[failure] = 0
            with self.assertRaises(ValueError):
                summarize(cells)

    def test_qualification_failure_stops_full_training_and_scheduler_does_not_pin(self):
        with patch.object(spec.sys, "argv", ["run", "--method", "retrace8", "--seed", "347", "--output", "/unused/result.json"]), \
                patch.object(spec.training, "run_cell", side_effect=RuntimeError("qualification failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                spec.main()
            self.assertEqual(run.call_count, 1)
        task = task_specification("test_trace", "retrace8", 347)
        self.assertEqual([p.split("/")[-1] for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(len(task["allowed_nodes"]), 6)


if __name__ == "__main__":
    unittest.main()
