"""Focused tests for the V29 zero-baseline quadratic treatment model."""

from __future__ import annotations

import json
import unittest

import numpy as np

from scripts.fit_freqduet_prefix_quadratic_value_model import (
    build_treatment_features,
    fit_zero_baseline_ridge,
    predict,
)
from scripts.fit_freqduet_prefix_value_model import select_rows
from tests.test_protocol_v6_v28_prefix_matrix import aggregated_labels_for_job


class V29QuadraticFeatureTest(unittest.TestCase):
    def test_actor_and_zero_offset_features_are_exactly_zero(self):
        labels = aggregated_labels_for_job()
        labels["actor_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["actor_action_json"]
        ]
        labels["candidate_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["candidate_action_json"]
        ]
        design, _ = build_treatment_features(
            labels, ["upper_state_000", "upper_state_001"]
        )
        identity = labels["candidate_method"].isin(
            ["actor", "actor_firstknot_0"]
        ).to_numpy()
        self.assertTrue(np.array_equal(
            design[identity], np.zeros_like(design[identity])
        ))

    def test_signed_quadratic_basis_distinguishes_interior_magnitude(self):
        labels = aggregated_labels_for_job()
        labels["actor_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["actor_action_json"]
        ]
        labels["candidate_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["candidate_action_json"]
        ]
        design, names = build_treatment_features(
            labels, ["upper_state_000", "upper_state_001"]
        )
        p15 = labels.index[labels["candidate_method"].eq(
            "actor_firstknot_p15"
        )][0]
        p30 = labels.index[labels["candidate_method"].eq(
            "actor_firstknot_p30"
        )][0]
        linear = names.index("positive")
        quadratic = names.index("positive_sq")
        self.assertAlmostEqual(design[p30, linear], 2.0 * design[p15, linear])
        self.assertAlmostEqual(
            design[p30, quadratic], 4.0 * design[p15, quadratic]
        )

    def test_ridge_preserves_exact_zero_treatment_prediction(self):
        x = np.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]])
        y = np.asarray([0.0, -0.1, 0.2])
        model = fit_zero_baseline_ridge(x, y, alpha=1.0)
        prediction = predict(model, x)
        self.assertEqual(float(prediction[0]), 0.0)
        self.assertTrue(np.isfinite(prediction).all())

    def test_quadratic_response_can_select_interior_action(self):
        labels = aggregated_labels_for_job()
        labels["actor_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["actor_action_json"]
        ]
        labels["candidate_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["candidate_action_json"]
        ]
        design, _ = build_treatment_features(
            labels, ["upper_state_000", "upper_state_001"]
        )
        signed = labels["candidate_offset_s"].to_numpy(dtype=np.float64) / 30.0
        positive = np.maximum(signed, 0.0)
        negative = np.maximum(-signed, 0.0)
        target = (
            -positive + 0.75 * np.square(positive)
            + 0.5 * negative + 0.5 * np.square(negative)
        )
        model = fit_zero_baseline_ridge(design, target, alpha=1e-9)
        selected = select_rows(labels, predict(model, design), guard_margin=0.0)
        self.assertEqual(
            selected.loc[0, "candidate_method"], "actor_firstknot_p15"
        )


if __name__ == "__main__":
    unittest.main()
