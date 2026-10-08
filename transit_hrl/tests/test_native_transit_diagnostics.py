import copy
import unittest

import numpy as np
import torch

from native_freqduet.lower.resac_lagrangian import GaussianPolicy
from native_freqduet.upper.resac_upper import BoundedGaussianPolicy
from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe, FEATURE_BLOCKS
from scripts import run_native_transit_diagnostics_stage147 as spec
from scripts import run_native_transit_routing_stage146 as routing
from scripts.analyze_native_transit_diagnostics_stage147 import summarize


class NativeDiagnosticsTest(unittest.TestCase):
    def policy(self, level):
        torch.manual_seed(7)
        return GaussianPolicy(33) if level == "lower" else BoundedGaussianPolicy(
            16, 1, action_low=[-120], action_high=[120])

    def test_passive_probe_preserves_actions_rng_and_weights(self):
        for level, width in (("upper", 16), ("lower", 33)):
            policy = self.policy(level)
            state = torch.ones(width)
            before = copy.deepcopy(policy.state_dict())
            torch.manual_seed(9)
            action = policy.get_action(state, deterministic=False)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(9)
            probe = NativePolicyProbe(policy, level)
            np.testing.assert_array_equal(probe.get_action(state), action)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            result = probe.summarize()
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(result["actor_calls"], 1)
            self.assertEqual(len(result["input_abs_mean"]), width)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, policy.state_dict()[key]))

    def test_neutral_probe_changes_command_not_proposal_or_rng(self):
        for level, width in (("upper", 16), ("lower", 33)):
            policy = self.policy(level)
            probe = NativePolicyProbe(policy, level, neutral=True)
            state = torch.ones(width)
            np.testing.assert_array_equal(probe.get_action(state, deterministic=True), 0)
            self.assertGreater(abs(probe.summarize()["proposal_mean_s"]), 0)

    def test_probe_measures_zero_band_effect_and_saturation(self):
        policy = self.policy("lower")
        with torch.no_grad():
            for p in policy.parameters():
                p.zero_()
            policy.fc1.weight[0, 29] = 1
            policy.fc2.weight[0, 0] = 1
            policy.mean.weight[0, 0] = 1
        probe = NativePolicyProbe(policy, "lower")
        state = torch.zeros(33)
        state[29] = 10
        probe.get_action(state, deterministic=True)
        result = probe.summarize()
        self.assertEqual(result["proposal_edge_fraction_1pct"], 1)
        self.assertAlmostEqual(result["zero_input_effect_s"]["dynamic_band"]["mean_abs"], 30)
        self.assertEqual(FEATURE_BLOCKS["upper"]["dynamic_band"], (1, 11))

    def test_only_correct_receives_physical_interventions_and_no_retraining(self):
        for method in spec.METHODS:
            self.assertEqual(spec.conditions(method, preflight=True), ("baseline",))
        self.assertEqual(spec.conditions("swapped_common", preflight=False), ("baseline",))
        self.assertEqual(len(spec.conditions("correct", preflight=False)), 3)
        self.assertEqual(spec.contract(False)["checkpoint_ep"], 299)
        self.assertEqual(spec.contract(False)["training_updates"], 0)

    def test_changed_baseline_or_exogenous_demand_stops_diagnostics(self):
        reference = {k: 1 for k in (*routing.METRICS, "simulation_end_time_s",
            "passengers_generated", "N_fleet", "passengers_unserved", "trips_completed", "ep_steps")}
        spec.check_episode(reference, reference, baseline=True)
        changed = dict(reference, ep_reward=2)
        with self.assertRaisesRegex(RuntimeError, "source reproduction"):
            spec.check_episode(changed, reference, baseline=True)
        spec.check_episode(changed, reference, baseline=False)
        with self.assertRaisesRegex(RuntimeError, "passengers_generated"):
            spec.check_episode(dict(reference, passengers_generated=2), reference, baseline=False)

    def cells(self):
        cells = {}
        for method in spec.METHODS:
            for root in spec.ROOTS:
                rows = []
                for condition in spec.conditions(method, preflight=False):
                    for scenario in spec.contract(False)["scenarios"]:
                        actors = {}
                        for level in ("upper", "lower"):
                            actors[level] = {
                                "proposal_mean_s": 1, "proposal_std_s": 2,
                                "proposal_edge_fraction_1pct": .5, "latent_mean_abs_p95": 3,
                                "input_abs_mean": [1] * (16 if level == "upper" else 33),
                                "zero_input_effect_s": {"dynamic_band": {"mean_abs": 4}},
                            }
                        rows.append({"condition": condition, "scenario": scenario,
                            "scene_seed": routing.evaluation_seeds(root, scenario, preflight=False)[0],
                            "actors": actors,
                            **{metric: root + (0 if condition == "baseline" else 2)
                               for metric in routing.METRICS}})
                cells[method, root] = {
                    "software_qualified": True, "baseline_reproduced": True,
                    "method": method, "seed": root, "protocol": spec.EXPERIMENT_PROTOCOL,
                    "contract": spec.contract(False), "training_updates": 0,
                    "native_steps": len(rows) * 61380, "evaluation": rows,
                }
        return cells

    def test_summary_preserves_root_pairing_and_descriptive_boundary(self):
        result = summarize(self.cells())
        self.assertIn("descriptive", result["stage"])
        self.assertEqual(result["native_steps"], 200 * 61380)
        for intervention in result["physical_interventions"].values():
            self.assertEqual(len(intervention["root_deltas"]), 8)
            self.assertEqual(intervention["mean_deltas"]["service_cost_restricted"], 2)
        self.assertEqual(result["baseline_actor_diagnostics"]["correct"]["lower"][
            "zero_input_mean_abs_effect_s"]["dynamic_band"], 4)

    def test_missing_or_duplicate_diagnostic_scene_is_rejected(self):
        for duplicate in (False, True):
            cells = self.cells()
            rows = cells["correct", spec.ROOTS[0]]["evaluation"]
            if duplicate:
                rows.append(copy.deepcopy(rows[0]))
            else:
                rows.pop()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                summarize(cells)


if __name__ == "__main__":
    unittest.main()
