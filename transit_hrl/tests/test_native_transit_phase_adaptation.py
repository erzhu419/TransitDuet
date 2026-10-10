import copy
import unittest

import numpy as np
import torch

from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe
from scripts import run_native_transit_phase_adaptation_stage154 as spec
from scripts.analyze_native_transit_phase_adaptation_stage154 import summarize
from scripts.submit_native_transit_phase_adaptation_stage154_scheduleurm import task_specification


class NativePhaseAdaptationTest(unittest.TestCase):
    def fixtures(self):
        sources, cells = {}, {}
        for root in spec.ROOTS:
            rows = []
            for scenario in spec.contract()["scenarios"]:
                for scene in spec.source_spec.scene_seeds(root, scenario, preflight=False):
                    baseline = {key: 1.0 for key in (*spec.METRICS, "passengers_unserved", "trips_completed", "ep_steps")}
                    baseline.update(simulation_end_time_s=61380, passengers_generated=100, N_fleet=12,
                        peak_fleet=14, scenario=scenario, scene_seed=scene,
                        actors={"upper": {"proposal_mean_s": 6.25, "actor_calls": 262}})
                    rows.extend(dict(baseline, condition=condition) for condition in spec.source_spec.CONDITIONS)
            source = {"protocol": spec.source_spec.EXPERIMENT_PROTOCOL, "contract": spec.source_spec.contract(False),
                "method": spec.METHOD, "seed": root, "software_qualified": True, "worker_preflight_passed": True,
                "updates": spec.source_spec.expected_updates(False), "critic_action_units": "unit",
                "weight_reg_mode": "physical_sum", "evaluation": rows}
            constants = spec.constant_actions(source)
            new_rows = []
            for original in rows:
                if original["condition"] != "baseline":
                    continue
                for condition in spec.CONDITIONS:
                    row = dict(original, condition=condition, fixed_action_s=constants.get(condition))
                    if condition != "baseline":
                        row["service_cost_restricted"] = 1.2
                        row["restricted_wait_horizon_min"] = .8
                        row["ep_reward"] = .5
                    new_rows.append(row)
            sources[root] = source
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "method": spec.METHOD, "seed": root, "software_qualified": True, "baseline_reproduced": True,
                "training_updates": 0, "constant_actions_s": constants, "native_steps": len(new_rows) * 61380,
                "evaluation": new_rows}
        return cells, sources

    def test_constant_mean_uses_actor_calls_once_across_source_scenes(self):
        source = {"evaluation": [
            {"condition": "baseline", "actors": {"upper": {"proposal_mean_s": 4, "actor_calls": 1}}},
            {"condition": "baseline", "actors": {"upper": {"proposal_mean_s": 8, "actor_calls": 3}}},
            {"condition": "neutral_upper", "actors": {"upper": {"proposal_mean_s": 0, "actor_calls": 100}}}]}
        self.assertEqual(spec.constant_actions(source), {"constant_mean": 7, "fixed5": 5, "fixed7": 7, "fixed9": 9})

    def test_learned_and_phase_gains_are_separate_not_an_oracle(self):
        cells, sources = self.fixtures()
        result = summarize(cells, sources)
        self.assertEqual(result["training_updates"], 0)
        self.assertEqual(result["native_steps"], 12276000)
        self.assertEqual(len(result["regime_root_means"]), 60)
        for row in result["learned_minus_constant"]["constant_mean"]:
            self.assertAlmostEqual(row["service_cost_restricted"], -.2)
            self.assertAlmostEqual(row["restricted_wait_horizon_min"], .2)
            self.assertAlmostEqual(row["ep_reward"], .5)
            self.assertEqual(row["fleet_cost_component"], 0)
        self.assertNotIn("best_fixed_command", result)

    def test_incomplete_changed_or_trained_results_do_not_merge(self):
        for failure in ("missing", "duplicate", "scene", "clock", "baseline", "command", "updates", "source"):
            with self.subTest(failure=failure):
                cells, sources = self.fixtures()
                cell = cells[spec.ROOTS[0]]
                rows = cell["evaluation"]
                if failure == "missing":
                    rows.pop()
                elif failure == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif failure == "scene":
                    rows[-1]["scene_seed"] += 1
                elif failure == "clock":
                    rows[1]["simulation_end_time_s"] -= 1
                elif failure == "baseline":
                    rows[0]["lower_action_mean"] += 1
                elif failure == "command":
                    rows[1]["fixed_action_s"] += 1
                elif failure == "updates":
                    cell["training_updates"] = 1
                else:
                    sources[spec.ROOTS[0]]["evaluation"].pop()
                with self.assertRaises((RuntimeError, ValueError)):
                    summarize(cells, sources)

    def test_fixed_command_calls_frozen_actor_without_mutating_input(self):
        class Policy(torch.nn.Module):
            def get_action(self, state, deterministic=False):
                self.calls += 1
                return np.asarray([float(np.sum(state))])
        policy = Policy()
        policy.calls = 0
        state = np.arange(16, dtype=np.float32)
        probe = NativePolicyProbe(policy, "upper", fixed_action_s=7)
        np.testing.assert_array_equal(probe.get_action(state, deterministic=True), [7])
        np.testing.assert_array_equal(state, np.arange(16, dtype=np.float32))
        self.assertEqual(policy.calls, 1)
        self.assertEqual(len(probe.states), 1)
        with self.assertRaisesRegex(RuntimeError, "training update"):
            spec.reject_training()

    def test_scheduler_reuses_server_checkpoint_without_staging_results(self):
        task = task_specification("test_phase", spec.ROOTS[0])
        self.assertEqual([path.split("/")[-1] for path in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(len(task["allowed_nodes"]), 6)
        self.assertEqual(task["cpu"], 1)
        self.assertIn("run_native_transit_phase_adaptation_stage154.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
