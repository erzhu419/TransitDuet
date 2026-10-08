import copy
import unittest

import numpy as np

from freq_hrl.domains.transit.native_routing import METHODS, NativeRoutingTracker
from native_freqduet.frequency.demand_frequency import DemandFrequencyTracker
from scripts import run_native_transit_routing_stage146 as spec
from scripts.analyze_native_transit_routing_stage146 import summarize


CFG = {
    "method": "harmonic", "bin_sec": 60, "od_features": True,
    "upper_mode": "low", "lower_mode": "high", "fourier_K": 1,
    "harmonic_period_s": 1440, "harmonic_prior_var": 0.01,
    "global_demand_norm": 1, "local_demand_norm": 1, "slope_norm": 1,
    "harmonic_prior": {"global": [1.0, 0.0, 0.0, 0.0]},
}


class NativeTransitRoutingTest(unittest.TestCase):
    def trackers(self):
        return {method: NativeRoutingTracker.from_config({**CFG, "routing": method}, 20)
                for method in METHODS}

    def test_correct_adapter_preserves_native_features(self):
        native = DemandFrequencyTracker.from_config(CFG, 20)
        routed = self.trackers()["correct"]
        for i in range(18):
            counts = {(1, True): i % 5, (2, False): 3}
            od = {(1, 2, True): i % 5, (2, 1, False): 3}
            for tracker in (native, routed):
                tracker.update(counts, od)
            np.testing.assert_array_equal(native.upper_features(), routed.upper_features())
            np.testing.assert_array_equal(native.lower_features(1, True), routed.lower_features(1, True))

    def test_only_actor_routing_changes_not_estimator_or_common_context(self):
        trackers = self.trackers()
        for i in range(18):
            for tracker in trackers.values():
                tracker.update({(1, True): i % 4}, {(1, 2, True): i % 4})
        native = trackers["correct"]
        for tracker in trackers.values():
            self.assertEqual(tracker.upper_feature_dim, 6)
            self.assertEqual(tracker.lower_feature_dim, 4)
            np.testing.assert_array_equal(tracker.global_state.theta, native.global_state.theta)
            np.testing.assert_array_equal(tracker.local_states[(1, True)].theta,
                                          native.local_states[(1, True)].theta)
            np.testing.assert_array_equal(tracker.upper_features()[2:], native.upper_features()[2:])
            self.assertEqual(tracker.summary(), native.summary())
        swapped = trackers["swapped_common"]
        np.testing.assert_array_equal(swapped.upper_features()[:2], native.upper_features("high")[:2])
        np.testing.assert_array_equal(swapped.lower_features(1, True), native.lower_features(1, True, "low"))

    def test_history_contains_closed_bins_and_zeros_for_absent_station(self):
        tracker = self.trackers()["raw_history_common"]
        for count in (1, 2, 3, 4, 5, 6):
            tracker.update({(1, True): count})
        # Two 60-second bins, each contains three 20-second arrival counts.
        np.testing.assert_array_equal(tracker.upper_features()[:2], [15, 9])
        np.testing.assert_array_equal(tracker.lower_features(1, True), [0, 0, 6, 15])
        for _ in range(3):
            tracker.update({})
        np.testing.assert_array_equal(tracker.lower_features(1, True), [0, 6, 15, 0])
        np.testing.assert_array_equal(tracker.lower_features(99, False), np.zeros(4))
        tracker.reset()
        np.testing.assert_array_equal(tracker.upper_features()[:2], [0, 0])
        np.testing.assert_array_equal(tracker.lower_features(1, True), np.zeros(4))

    def test_all_routing_prefixes_are_causal(self):
        for method in METHODS:
            def run(values):
                tracker = NativeRoutingTracker.from_config({**CFG, "routing": method}, 60)
                result = []
                for value in values:
                    tracker.update({(1, True): value})
                    result.append(np.concatenate([tracker.upper_features(), tracker.lower_features(1, True)]))
                return np.asarray(result)
            np.testing.assert_array_equal(run([1, 2, 3]), run([1, 2, 3, 1000])[:3])

    def test_config_changes_only_routing_between_methods(self):
        base = {"frequency": copy.deepcopy(CFG), "env": {}, "seed": 0,
                "upper": {}, "lower": {}, "coupling": {"upper_warmup_eps": 30}}
        configs = [spec.configure(base, method, 101, preflight=False) for method in METHODS]
        for cfg in configs:
            cfg["frequency"].pop("routing")
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])
        self.assertFalse(configs[0]["env"]["allow_early_finish"])
        self.assertEqual(configs[0]["env"]["evaluation_end_time_s"], 60000)
        self.assertNotIn("allow_early_finish", base["env"])

    def test_evaluation_seeds_are_disjoint_from_training_and_between_roots(self):
        seen = set()
        for root in spec.ROOTS:
            training = {root * 1000003 + i for i in range(300)}
            for scenario in spec.SCENARIOS:
                seeds = set(spec.evaluation_seeds(root, scenario, preflight=False))
                self.assertFalse(seeds & (seen | training))
                seen.update(seeds)
        self.assertEqual(len(seen), 8 * 5 * 4)


class NativeRoutingAnalysisTest(unittest.TestCase):
    def cells(self, preflight=False):
        cells = {}
        roots = spec.ROOTS[:1] if preflight else spec.ROOTS
        for method in METHODS:
            for root in roots:
                evaluation = []
                for scenario in spec.contract(preflight)["evaluation_scenarios"]:
                    for seed in spec.evaluation_seeds(root, scenario, preflight=preflight):
                        evaluation.append({
                            "scenario": scenario, "scene_seed": seed,
                            "simulation_end_time_s": spec.contract(preflight)["training_clock_s"],
                            "passengers_generated": 100, "N_fleet": 12,
                            **{metric: float(root) + (0 if method == "correct" else 2)
                               for metric in spec.METRICS},
                        })
                cells[method, root] = {
                    "passed": True, "method": method, "seed": root,
                    "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(preflight),
                    "actor_dims": {"upper": 16, "lower": 33}, "parameter_counts": {"upper": 100, "lower": 200},
                    "updates": {"upper": 2, "lower": 4}, "native_steps": 1000,
                    "training_demand_counts": [100, 101], "evaluation": evaluation,
                }
        return cells

    def test_ci_clusters_by_optimizer_not_evaluation_episode(self):
        result = summarize(self.cells(), preflight=False)
        self.assertEqual(len(result["optimizer_roots"]), 8)
        for comparison in result["comparisons"].values():
            self.assertEqual(len(comparison["root_deltas"]), 8)
            self.assertEqual(comparison["primary_mean_delta"], -2)
            self.assertEqual(comparison["family_adjusted_primary_ci"], [-2, -2])
            self.assertEqual(comparison["primary_status"], "development_supported")

    def test_preflight_never_reports_a_performance_ci(self):
        result = summarize(self.cells(preflight=True), preflight=True)
        self.assertTrue(result["software_qualified"])
        self.assertNotIn("comparisons", result)

    def test_unmatched_capacity_or_exogenous_demand_is_rejected(self):
        for change in ("parameter_counts", "training_demand_counts", "evaluation"):
            cells = self.cells()
            bad = cells["swapped_common", spec.ROOTS[0]]
            if change == "evaluation":
                bad["evaluation"][0]["passengers_generated"] += 1
            elif change == "training_demand_counts":
                bad[change][0] += 1
            else:
                bad[change]["upper"] += 1
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                summarize(cells, preflight=False)


if __name__ == "__main__":
    unittest.main()
