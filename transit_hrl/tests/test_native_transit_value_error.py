import unittest

from scripts.run_native_transit_value_error_stage189 import diagnose, summarize


class ValueErrorTest(unittest.TestCase):
    def inputs(self):
        labels = [{"actual_cost_delta": c, "wait_delta_min": w} for c, w in
            ((0., 0.), (.2, .1), (-.1, -.1), (.1, -.1), (0., 0.), (0., 0.))]
        row = {"root": 397, "scene_seed": 1, "probe_index": 8, "original_choice": 0, "labels": labels}
        prediction = {"selected": 1, "predicted_heads": [[c, 0.] for c in (0., -.3, -.2, -.25, .1, .1)],
            "predicted_wait_delta_min": [0., -.1, -.1, -.1, 0., 0.]}
        return prediction, row

    def test_oracle_wait_isolates_ranking_without_changing_predictions(self):
        p, row = self.inputs()
        d = diagnose(p, row)
        self.assertEqual(d["arms"]["predicted"]["selected"], 1)
        self.assertEqual(d["arms"]["oracle_wait"]["selected"], 3)
        self.assertGreater(d["arms"]["oracle_wait"]["cost_delta"], 0)
        self.assertLessEqual(d["arms"]["oracle_wait"]["wait_delta_min"], 0)
        self.assertEqual(d["arms"]["oracle_cost"]["selected"], 2)
        self.assertEqual(d["arms"]["oracle_both"]["selected"], 2)
        self.assertEqual(p["selected"], 1)

    def test_selection_bias_and_summary_use_physical_units(self):
        p, row = self.inputs()
        d = diagnose(p, row)
        self.assertAlmostEqual(d["selected_cost_bias"], -.5)
        self.assertAlmostEqual(d["selected_wait_bias_min"], -.2)
        s = summarize([d, d])
        self.assertEqual(s["arms"]["predicted"]["positive_wait_choices"], 2)
        self.assertEqual(s["arms"]["oracle_wait"]["positive_wait_choices"], 0)

    def test_mismatch_cannot_silently_change_the_cached_selector(self):
        p, row = self.inputs()
        p["selected"] = 2
        with self.assertRaises(ValueError): diagnose(p, row)


if __name__ == "__main__":
    unittest.main()
