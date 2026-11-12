import json
import unittest

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.pointmaze_compact_plan_validity import _quadratic_features
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_stage9_scale_ablation import METHODS, VIEWS, fit_heads, readout


class ScaleAblationTest(unittest.TestCase):
    def test_equal_capacity_and_serialized_causal_score(self):
        names = sorted(set().union(*map(set, VIEWS.values())))
        rng = np.random.default_rng(15)
        raw = rng.normal(size=(72, len(names)))
        rows = [{"seed": 100 + i // 9, "feature_names": names, "causal_features": x.tolist(),
                 "renew_ise_advantage": float(x[0] + .5 * x[-1])} for i, x in enumerate(raw)]
        ledger = ComputeLedger()
        with ledger.recording():
            models = fit_heads(rows, {"views": VIEWS, "ridge_df": 8., "threshold_quantile": .75})
        self.assertEqual(ledger.summary()["counts"]["ridge_fit_calls"], 27)
        for name, record in json.loads(json.dumps(models)).items():
            rule = TimingRule(**record)
            self.assertEqual(rule.model["coefficient_count"], 91)
            self.assertAlmostEqual(rule.model["effective_df_slopes"], 8.)
            x, _ = _quadratic_features(raw[:1, [names.index(n) for n in VIEWS[name]]], VIEWS[name])
            z = (x[0] - rule.model["feature_mean"]) / rule.model["feature_scale"]
            self.assertAlmostEqual(rule.score(tuple(names), raw[0], 0., 1.),
                                   rule.model["weights"][0] + z @ rule.model["weights"][1:])

    def test_mixed_scale_requires_single_scale_contrast(self):
        cells = [{"optimizer_seed": i, "evaluation_seeds": [2 * i, 2 * i + 1]} for i in range(8)]
        plan = {"cells": cells, "steps_per_root": 9600, "minimum_relative_gain": .02, "scope": "fixture"}
        rows = []
        for c in cells:
            panel = {m: {"mean_ise": .9 if m in ("mixed_scale", "single_scale") else 1.} for m in METHODS}
            rows.append({**c, "status": "complete", "panels": {p: panel for p in ("all", "first", "last")},
                         "measured_cost": {"counts": {"environment_steps": 9600}}})
        summary = readout(plan, rows)
        self.assertFalse(summary["mixed_scale_supported"])
        self.assertTrue(summary["history_supported"])
        self.assertFalse(summary["original_confirmation_released"])
        with self.assertRaises(ValueError):
            readout(plan, rows[::-1])
