"""Focused tests for the V32 pairwise composite-service planner."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.audit_protocol_v6_v28_prefix_common import (
    DECISION_INDICES as V28_DECISIONS,
)
from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CONFIRMATION_DECISION_INDICES as V30_CONFIRMATION_DECISIONS,
    CONFIRMATION_POLICY_SEEDS as V30_CONFIRMATION_POLICIES,
    CONFIRMATION_SCENARIO_SEEDS as V30_CONFIRMATION_SCENARIOS,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
)
from scripts.audit_protocol_v6_v31_pairwise_common import (
    CONFIRMATION_DECISION_INDICES as V31_CONFIRMATION_DECISIONS,
    CONFIRMATION_POLICY_SEEDS as V31_CONFIRMATION_POLICIES,
    CONFIRMATION_SCENARIO_SEEDS as V31_CONFIRMATION_SCENARIOS,
)
from scripts.audit_protocol_v6_v32_pairwise_composite_common import (
    CONFIRMATION_DECISION_INDICES,
    CONFIRMATION_POLICY_SEEDS,
    CONFIRMATION_SCENARIO_SEEDS,
    SERVICE_REFERENCE_METHOD,
    confirmation_is_fresh_from_v31,
)
from scripts.fit_freqduet_v32_pairwise_composite_value_model import (
    fit_service_pipeline,
    predict_service,
    select_pairwise_rows,
)
from tests.test_protocol_v6_v31_pairwise_safe_value import (
    STATE_COLS,
    v31_contexts,
)


class V32RosterTest(unittest.TestCase):
    def test_confirmation_roster_is_disjoint_from_all_prefix_rosters(self):
        self.assertTrue(confirmation_is_fresh_from_v31())
        self.assertTrue(set(CONFIRMATION_POLICY_SEEDS).isdisjoint(
            DISCOVERY_TRAIN_SEEDS + V30_CONFIRMATION_POLICIES
            + V31_CONFIRMATION_POLICIES
        ))
        self.assertTrue(set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            DISCOVERY_SCENARIO_SEEDS + V30_CONFIRMATION_SCENARIOS
            + V31_CONFIRMATION_SCENARIOS
        ))
        prior_decisions = (
            V28_DECISIONS + DISCOVERY_DECISION_INDICES
            + V30_CONFIRMATION_DECISIONS + V31_CONFIRMATION_DECISIONS
        )
        self.assertTrue(set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            prior_decisions
        ))


class V32SelectorTest(unittest.TestCase):
    def test_selector_keeps_p30_without_predicted_composite_gain(self):
        labels = v31_contexts(count=1)
        prediction = np.full(len(labels), 0.1, dtype=np.float64)
        prediction[labels["candidate_method"].eq(
            SERVICE_REFERENCE_METHOD
        ).to_numpy()] = 0.0
        selected = select_pairwise_rows(
            labels, prediction, service_margin=0.0
        )
        self.assertEqual(
            selected.loc[0, "candidate_method"], SERVICE_REFERENCE_METHOD
        )

    def test_selector_uses_best_material_composite_gain(self):
        labels = v31_contexts(count=1)
        values = {
            "actor": 0.2,
            "actor_firstknot_m30": -0.3,
            "actor_firstknot_m15": -0.1,
            "actor_firstknot_p15": -0.2,
            SERVICE_REFERENCE_METHOD: 0.0,
        }
        prediction = labels["candidate_method"].map(values).to_numpy(
            dtype=np.float64
        )
        selected = select_pairwise_rows(
            labels, prediction, service_margin=0.0001
        )
        self.assertEqual(
            selected.loc[0, "candidate_method"], "actor_firstknot_m30"
        )

    def test_pairwise_quadratic_pipeline_can_choose_interior(self):
        labels = v31_contexts(count=6)
        signed = labels["candidate_offset_s"].to_numpy(dtype=np.float64) / 30.0
        positive = np.maximum(signed, 0.0)
        negative = np.maximum(-signed, 0.0)
        labels["episode_service_cost_restricted_delta_vs_actor"] = (
            -positive + 0.75 * np.square(positive)
            + 0.5 * negative + 0.5 * np.square(negative)
        )
        pipeline, design, names = fit_service_pipeline(
            labels,
            STATE_COLS,
            np.ones(len(labels), dtype=bool),
            rank=0,
            alpha=1e-9,
        )
        self.assertEqual(len(names), 4)
        p30 = labels["candidate_method"].eq(
            SERVICE_REFERENCE_METHOD
        ).to_numpy()
        predicted = predict_service(pipeline, design)
        self.assertTrue(np.array_equal(
            predicted[p30], np.zeros(p30.sum(), dtype=np.float64)
        ))
        first = labels["decision_index"].eq(1).to_numpy()
        selected = select_pairwise_rows(
            labels.loc[first].reset_index(drop=True),
            predicted[first],
            service_margin=0.0,
        )
        self.assertEqual(
            selected.loc[0, "candidate_method"], "actor_firstknot_p15"
        )


if __name__ == "__main__":
    unittest.main()
