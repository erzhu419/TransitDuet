"""Focused tests for the V29 fold-local low-rank quadratic model."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from scripts.fit_freqduet_prefix_lowrank_quadratic_value_model import (
    fit_context_projection,
    fit_pipeline,
    predict_pipeline,
    transform_features,
)
from scripts.fit_freqduet_prefix_value_model import select_rows
from tests.test_protocol_v6_v28_prefix_matrix import aggregated_labels_for_job


STATE_COLS = ["upper_state_000", "upper_state_001"]
CONTEXT_VALUES = [
    (0.0, 0.0, 1.0, 2.0),
    (1.0, 0.0, 1.2, 3.0),
    (0.0, 1.0, 2.0, 2.0),
    (1.0, 1.0, 3.0, 4.0),
    (2.0, -1.0, 0.0, 5.0),
    (-1.0, 2.0, 4.0, 1.0),
]


def prepared_contexts(count: int = 6) -> pd.DataFrame:
    parts = []
    for index in range(count):
        state_0, state_1, actor_0, actor_1 = CONTEXT_VALUES[index]
        labels = aggregated_labels_for_job().copy()
        labels["decision_index"] = index + 1
        labels["dispatch_index"] = index + 1
        labels["upper_state_000"] = state_0
        labels["upper_state_001"] = state_1
        actor = [actor_0, actor_1]
        labels["actor_action_json"] = json.dumps(actor)
        labels["candidate_action_json"] = [
            json.dumps([
                actor[0] + float(offset) if method not in {
                    "actor", "actor_firstknot_0"
                } else actor[0],
                actor[1],
            ])
            for method, offset in zip(
                labels["candidate_method"], labels["candidate_offset_s"]
            )
        ]
        parts.append(labels)
    labels = pd.concat(parts, ignore_index=True)
    labels["actor_action"] = [
        np.asarray(json.loads(value), dtype=np.float64)
        for value in labels["actor_action_json"]
    ]
    labels["candidate_action"] = [
        np.asarray(json.loads(value), dtype=np.float64)
        for value in labels["candidate_action_json"]
    ]
    return labels


class V29LowRankQuadraticTest(unittest.TestCase):
    def test_rank_four_has_twenty_features_and_zero_identity(self):
        labels = prepared_contexts()
        train_mask = np.ones(len(labels), dtype=bool)
        projection = fit_context_projection(
            labels, STATE_COLS, train_mask, rank=4
        )
        design, names = transform_features(labels, STATE_COLS, projection)
        self.assertEqual(design.shape, (len(labels), 20))
        self.assertEqual(len(names), 20)
        identity = labels["candidate_method"].isin(
            ["actor", "actor_firstknot_0"]
        ).to_numpy()
        self.assertTrue(np.array_equal(
            design[identity], np.zeros_like(design[identity])
        ))

    def test_projection_normalization_excludes_heldout_context(self):
        labels = prepared_contexts(count=3)
        train_mask = labels["decision_index"].le(2).to_numpy()
        projection = fit_context_projection(
            labels, STATE_COLS, train_mask, rank=0
        )
        expected = np.asarray([0.5, 0.0, 1.1 / 60.0, 2.5 / 60.0])
        np.testing.assert_allclose(projection["context_mean"], expected)
        self.assertEqual(projection["fitted_actor_contexts"], 2)

    def test_rank_zero_quadratic_can_select_interior_action(self):
        labels = prepared_contexts(count=2)
        train_mask = np.ones(len(labels), dtype=bool)
        signed = labels["candidate_offset_s"].to_numpy(dtype=np.float64) / 30.0
        positive = np.maximum(signed, 0.0)
        negative = np.maximum(-signed, 0.0)
        labels["episode_service_cost_restricted_delta_vs_actor"] = (
            -positive + 0.75 * np.square(positive)
            + 0.5 * negative + 0.5 * np.square(negative)
        )
        pipeline, design, names = fit_pipeline(
            labels,
            STATE_COLS,
            train_mask,
            rank=0,
            alpha=1e-9,
        )
        self.assertEqual(len(names), 4)
        first = labels[labels["decision_index"].eq(1)].reset_index(drop=True)
        first_design = design[labels["decision_index"].eq(1).to_numpy()]
        selected = select_rows(
            first,
            predict_pipeline(pipeline, first_design),
            guard_margin=0.0,
        )
        self.assertEqual(
            selected.loc[0, "candidate_method"], "actor_firstknot_p15"
        )


if __name__ == "__main__":
    unittest.main()
