import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_frozen_prefix_stage287 as spec
from scripts import submit_native_transit_frozen_prefix_stage287_scheduleurm as submit

learning = spec.learning


def profile(raw=0, **choices):
    return {"choices": {v: [choices.get(v, raw)]*6 for v in learning.VARIANTS}}


class FrozenPrefixTest(unittest.TestCase):
    def test_actual_within_profile_divergence_not_cross_profile_variation(self):
        self.assertEqual(learning.active_union([profile(0), profile(1)], [2]*6), [[]]*6)
        self.assertEqual(learning.active_union([profile(0, observed_level=1), profile(3)], [2]*6), [[0, 1, 2]]*6)

    def test_mc_pair_covariance_fixed_profile_average_and_all_state_denominator(self):
        p = profile(0, action_level=1, observed_level=2)
        for v in learning.VARIANTS:
            p["choices"][v][1:] = [0]*5
        labels = {(0, d): {0: [0.]*4, 1: [d+1.]*4, 2: [d+2.]*4} for d in range(4)}
        mean, covariance = learning.prefix_effects([p], labels, 4)
        duplicate, repeated = learning.prefix_effects([p]*32, labels, 4)
        np.testing.assert_allclose(mean, duplicate)
        np.testing.assert_allclose(covariance, repeated)
        self.assertAlmostEqual(mean[4, 0], 3.5/6)
        # Both actions share the same draw noise; their difference is constant.
        np.testing.assert_allclose(covariance[4, 4]+covariance[3, 3]-2*covariance[4, 3], 0., atol=1e-16)
        zero, variance = learning.prefix_effects([profile()], {}, 4)
        self.assertFalse(zero.any())
        self.assertFalse(variance.any())

    def test_prefix_unit_bootstrap_and_bridge_mc_only(self):
        records = []
        for root in (397, 401):
            for scene in range(8):
                mean = np.zeros((5, 4))
                mean[4] = -scene-1.
                records.append({"root": root, "scenario": "low_noise", "profiles": 32,
                    "mean": mean.tolist(), "MC_covariance": np.zeros((5, 5, 4)).tolist()})
        result = learning.summarize_prefixes(records, new=True, bootstrap=200)
        self.assertEqual(result["prefixes"], 16)
        self.assertEqual(result["states"], 96)
        self.assertEqual(result["contrasts"]["observed_level_vs_uniform"]["mean"]["cost"], -4.5)
        self.assertLess(result["contrasts"]["observed_level_vs_uniform"]["prefix_CI95"]["cost"][1], 0.)
        bridge = learning.summarize_prefixes(records[:1], new=False)
        self.assertNotIn("prefix_CI95", bridge["contrasts"]["vs_raw_observed_level"])

    def test_inference_without_labels_preserves_old_path_and_new_raw_risk(self):
        n = 2
        data = {"originals": np.zeros(n, int), "heads": np.zeros((n, 6, 2)),
            "components": np.zeros((n, 6, 2)), "raw_logits": -np.ones((n, 6)),
            "x": np.zeros((n, 6, 198)), "raw_choices": np.ones(n, int)}
        data["heads"][:, 1:, 0] = .002
        stats = {"center": np.zeros(8), "scale": np.ones(8), "weights": np.r_[-.004, np.zeros(8)][:, None]}
        geometry = {"radius": .01, "maps": {"linear": stats, "observed_context_df": {"context_stats": stats}}}
        package = {"reference": {}, "basis": np.eye(198, 8), "stage247_model": {"old": True},
            "stage247_selection": "frozen", "banks": [{"bank": 0, "geometry": geometry,
                "level": {"maps": {k: stats for k in ("action_level", "observed_level")}}}]}
        old = np.full(n, 3)
        base = SimpleNamespace(causal_data=lambda rows, ref: copy.deepcopy(data),
            candidates=lambda d, model: {"frozen": {"chosen": old}})
        with patch.object(learning.level.service.state, "observed_context", return_value=np.zeros((n, 8))):
            result = learning.infer([{}, {}], package, SimpleNamespace(learning=SimpleNamespace(base=base)))[0]
        np.testing.assert_array_equal(result["choices"]["stage247"], old)
        for k in ("uniform", "action_level", "observed_level"):
            np.testing.assert_array_equal(result["choices"][k], [1, 1])

    def test_scene_future_pilot_disjointness_and_scheduler_unpinned(self):
        main, pilot = spec.prefixes(), spec.prefixes(pilot=True)
        self.assertEqual(len(main), 90)
        self.assertEqual(len(pilot), 20)
        new = {spec.scene_seed(*k) for k in main if k[0] == "new"}
        pilots = {spec.scene_seed(*k) for k in pilot if k[0] == "pilot_new"}
        self.assertFalse(new & pilots)
        futures = [spec.future_seed(*k, p, d) for k in main for p in spec.PROBES
            for d in (range(1, 33) if k[0] == "bridge" else range(64))]
        self.assertEqual(len(futures), len(set(futures)))
        self.assertFalse(set(futures)&{spec.future_seed(*k, 22, 0) for k in pilot})
        self.assertEqual([d for chunk in spec.chunks("bridge") for d in chunk], list(range(1, 33)))
        task = submit.task_specification("unit_frozen_prefix", "pilot", pilot[0])
        self.assertEqual(task["cpu"], 1)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertNotIn("fit", task["cmd"])
        self.assertEqual(json.loads(json.dumps(spec.contract())), spec.contract())


if __name__ == "__main__":
    unittest.main()
