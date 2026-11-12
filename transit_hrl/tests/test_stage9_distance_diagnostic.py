import unittest

from scripts.run_stage9_distance_diagnostic import summarize


class DistanceDiagnosticTest(unittest.TestCase):
    def test_all_roots_paired_and_no_confirmation_release(self):
        cells = [{"optimizer_seed": i, "diagnostic_seeds": [10 * i, 10 * i + 1]} for i in range(8)]
        plan = {"cells": cells, "steps_per_root": 4800, "scope": "development",
                "development_signal_minimum_relative_gain": .02}
        records = [{"status": "complete", "optimizer_seed": c["optimizer_seed"], "latency": {},
            "sustained_failure_counts": {}, "measured_cost": {"counts": {"environment_steps": 4800}},
            "paths": [{"seed": s, "history": .9, "threshold": 1., "schedule_differs": True}
                      for s in c["diagnostic_seeds"]]} for c in cells]
        result = summarize(plan, records)
        self.assertEqual(result["paired_ise_gain"]["n"], 8)
        self.assertTrue(result["development_signal"])
        self.assertFalse(result["original_confirmation_released"])
        for r in records:
            r["paths"][1]["history"] = 1.01
        self.assertFalse(summarize(plan, records)["development_signal"])
        with self.assertRaises(ValueError):
            summarize(plan, records[:-1])
