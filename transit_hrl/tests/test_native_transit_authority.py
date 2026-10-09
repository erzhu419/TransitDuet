import copy
import unittest

import numpy as np
import torch

from native_freqduet.lower.state_encoder import PhysicalLowerStateEncoder
from native_freqduet.upper.interval_credit import UpperIntervalOutcomeTracker
from native_freqduet.upper.resac_upper import BoundedGaussianPolicy
from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe
from scripts import run_native_transit_authority_stage148 as spec
from scripts.analyze_native_transit_authority_stage148 import summarize
from scripts.submit_native_transit_authority_stage148_scheduleurm import task_specification


class NativeAuthorityTest(unittest.TestCase):
    def test_legacy_unchanged_and_factors_are_separate(self):
        base = {"frequency": {}, "env": {}, "coupling": {}, "upper": {}, "lower": {}}
        before = copy.deepcopy(base)
        legacy = spec.configure(base, "legacy", 217, preflight=False)
        self.assertEqual(legacy, spec.routing.configure(base, "correct", 217, preflight=False))
        physical = spec.configure(base, "physical_lower", 217, preflight=False)
        credit = spec.configure(base, "service_credit", 217, preflight=False)
        both = spec.configure(base, "physical_lower_service_credit", 217, preflight=False)
        self.assertEqual(physical["upper"], legacy["upper"])
        self.assertEqual(credit["lower"], legacy["lower"])
        self.assertEqual(both["upper"], credit["upper"])
        self.assertEqual(both["lower"], physical["lower"])
        self.assertEqual(base, before)
        self.assertEqual(both["lower"]["state_encoder"]["input_schema"], "explicit_target_v2")
        self.assertEqual(both["upper"]["interval_credit"]["weights"]["headway"], 0)

    def test_physical_encoder_keeps_band_and_changes_goal_slots(self):
        encoder = PhysicalLowerStateEncoder(29, 21, 14, 60, input_schema="explicit_target_v2")
        raw = np.array([11, 10, 7, 1, 360, 360, 90, 360, *([7.5] * 21), .01, .02, .03, .04], dtype=np.float32)
        encoded = encoder.encode(raw)
        self.assertEqual(encoded.size, 33)
        np.testing.assert_array_equal(encoded[29:], raw[29:])
        self.assertAlmostEqual(float(encoded[0]), .6)
        self.assertAlmostEqual(float(encoded[4]), 1)
        raw[7] = 420
        changed = encoder.encode(raw)
        self.assertAlmostEqual(float(changed[0]), .7)
        self.assertAlmostEqual(float(changed[4]), 360 / 420)
        self.assertAlmostEqual(float(changed[7]), -60 / 420)
        self.assertEqual(raw[0], 11)

    def test_physical_credit_cannot_gain_from_changing_headway_target(self):
        rewards = []
        for target in (240, 360, 480):
            tracker = UpperIntervalOutcomeTracker(enabled=True, reward_scale=100, headway_weight=0)
            tracker.begin("__legacy_global__", 0)
            tracker.record_step(dt_s=60, waiting_by_direction={True: 10, False: 20},
                fleet_by_direction={True: 7, False: 7}, n_fleet_target=12,
                headway_events=[{"direction": True, "headway_s": 360, "target_headway_s": target}])
            score = tracker.score(tracker.close("__legacy_global__", 60),
                passengers_generated=100, episode_headway_samples=1, episode_duration_s=60, n_fleet_target=12)
            rewards.append(score["reward"])
        np.testing.assert_array_equal(rewards, [rewards[0]] * 3)
        row = {"upper_interval_reward_sum": -3.0, "upper_interval_wait_cost_sum": .01,
               "upper_interval_fleet_cost_sum": .02, "upper_interval_coverage_mean": 1}
        spec.credit_row(row, "service_credit")
        with self.assertRaisesRegex(RuntimeError, "physical wait/fleet"):
            spec.credit_row(dict(row, upper_interval_reward_sum=1), "service_credit")

    def test_fixed_upper_command_preserves_rng_and_proposals(self):
        torch.manual_seed(7)
        policy = BoundedGaussianPolicy(16, 1, action_low=[-120], action_high=[120])
        for fixed in (-60, 60):
            torch.manual_seed(9)
            policy.get_action(torch.ones(16))
            rng = torch.get_rng_state().clone()
            torch.manual_seed(9)
            probe = NativePolicyProbe(policy, "upper", fixed_action_s=fixed)
            np.testing.assert_array_equal(probe.get_action(torch.ones(16)), [fixed])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(probe.summarize()["fixed_action_s"], fixed)

    def cells(self, preflight=False):
        contract = spec.contract(preflight)
        cells = {}
        for index, method in enumerate(spec.METHODS):
            for root in contract["roots"]:
                rows = [{"condition": condition, "scenario": scenario,
                    "scene_seed": spec.scene_seed(root, scenario), "N_fleet": 12,
                    "simulation_end_time_s": contract["training_clock_s"], "passengers_generated": 100,
                    "actors": {"lower": {"zero_input_effect_s": {"dynamic_band": {"mean_abs": .1}}}},
                    **{metric: index + (0 if condition == "baseline" else 2) for metric in
                       (*spec.routing.METRICS, "lower_action_mean", "upper_delta_mean")}}
                    for condition in spec.CONDITIONS for scenario in contract["scenarios"]]
                cells[method, root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
                    "software_qualified": True, "method": method, "seed": root,
                    "actor_dims": contract["actor_dims"], "parameter_counts": {"upper": 1, "lower": 1},
                    "updates": {"upper": (contract["train_episodes"]-contract["upper_warmup"]) * (2 if preflight else 10),
                                "lower": contract["train_episodes"] * (2 if preflight else 30)},
                    "actor_change_max_abs": {"upper": .1, "lower": .2},
                    "native_steps": (contract["train_episodes"] + len(rows)) * contract["training_clock_s"],
                    "training_demand_counts": [100] * contract["train_episodes"], "evaluation": rows}
        return cells

    def test_descriptive_root_pairing_and_preflight_budget(self):
        result = summarize(self.cells(), preflight=False)
        self.assertIn("descriptive", result["stage"])
        self.assertEqual(result["native_steps"], 2600 * 61380)
        self.assertEqual(result["baseline_minus_legacy_root_deltas"]["physical_lower"][0]["service_cost_restricted"], 1)
        for method in spec.METHODS:
            self.assertEqual(result["physical_intervention_minus_baseline_root_deltas"][method][
                "neutral_upper"][0]["service_cost_restricted"], 2)
        preflight = summarize(self.cells(True), preflight=True)
        self.assertEqual(preflight["native_steps"], 28 * 5400)
        self.assertNotIn("baseline_minus_legacy_root_deltas", preflight)

    def test_missing_duplicate_demand_or_update_mismatch_rejected(self):
        for failure in ("missing", "duplicate", "demand", "update"):
            cells = self.cells()
            cell = cells["physical_lower", 217]
            if failure == "missing":
                cell["evaluation"].pop()
            elif failure == "duplicate":
                cell["evaluation"].append(copy.deepcopy(cell["evaluation"][0]))
            elif failure == "demand":
                cell["evaluation"][0]["passengers_generated"] = 101
            else:
                cell["updates"]["upper"] = 0
            with self.assertRaises(ValueError):
                summarize(cells, preflight=False)

    def test_scheduler_stages_only_code_and_all_cpu_nodes_are_eligible(self):
        for method in spec.METHODS:
            task = task_specification("test_authority", method, 217, preflight=True)
            self.assertEqual([p.split("/")[-1] for p in task["stage_input_paths"]],
                             ["scripts", "freq_hrl", "native_freqduet"])
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("run_native_transit_authority_stage148.py", task["cmd"])
            self.assertIn(f"--method {method}", task["cmd"])


if __name__ == "__main__":
    unittest.main()
