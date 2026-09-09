"""Focused tests for the V31 pairwise multi-head value planner."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CONFIRMATION_DECISION_INDICES as V30_CONFIRMATION_DECISIONS,
    CONFIRMATION_POLICY_SEEDS as V30_CONFIRMATION_POLICIES,
    CONFIRMATION_SCENARIO_SEEDS as V30_CONFIRMATION_SCENARIOS,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
)
from scripts.audit_protocol_v6_v31_pairwise_common import (
    CONFIRMATION_DECISION_INDICES,
    CONFIRMATION_POLICY_SEEDS,
    CONFIRMATION_SCENARIO_SEEDS,
    confirmation_is_disjoint_from_development,
)
from scripts.fit_freqduet_v31_pairwise_safe_value_model import (
    SERVICE_REFERENCE_METHOD,
    contrast_to_method,
    fit_context_projection,
    fit_pipeline,
    predict_pipeline,
    select_constrained_rows,
    transform_features,
)
from tests.test_protocol_v6_v30_expanded_prefix import prepared_contexts


STATE_DIM = 10
STATE_COLS = [f"upper_state_{index:03d}" for index in range(STATE_DIM)]


def v31_contexts(count: int = 12):
    labels = prepared_contexts(count=count)
    labels["upper_state_dim"] = STATE_DIM
    context_index = labels["decision_index"].to_numpy(dtype=np.float64)
    for column_index, column in enumerate(STATE_COLS):
        labels[column] = np.sin(
            context_index * float(column_index + 1) / 3.0
        )
    return labels.reset_index(drop=True)


def predictions(labels, *, service, headway, unserved):
    methods = labels["candidate_method"].astype(str)
    return {
        "service_vs_p30": methods.map(service).to_numpy(dtype=np.float64),
        "headway_vs_actor": methods.map(headway).to_numpy(dtype=np.float64),
        "unserved_vs_actor": methods.map(unserved).to_numpy(dtype=np.float64),
    }


class V31RosterTest(unittest.TestCase):
    def test_confirmation_roster_is_fresh_against_v30(self):
        self.assertTrue(confirmation_is_disjoint_from_development())
        self.assertTrue(set(CONFIRMATION_POLICY_SEEDS).isdisjoint(
            DISCOVERY_TRAIN_SEEDS
        ))
        self.assertTrue(set(CONFIRMATION_POLICY_SEEDS).isdisjoint(
            V30_CONFIRMATION_POLICIES
        ))
        self.assertTrue(set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            DISCOVERY_SCENARIO_SEEDS
        ))
        self.assertTrue(set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            V30_CONFIRMATION_SCENARIOS
        ))
        self.assertTrue(set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            DISCOVERY_DECISION_INDICES
        ))
        self.assertTrue(set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            V30_CONFIRMATION_DECISIONS
        ))


class V31FeatureContractTest(unittest.TestCase):
    def test_rank_eight_is_36d_with_exact_actor_and_p30_zeros(self):
        labels = v31_contexts()
        mask = np.ones(len(labels), dtype=bool)
        projection = fit_context_projection(
            labels, STATE_COLS, mask, rank=8
        )
        absolute, names = transform_features(
            labels, STATE_COLS, projection
        )
        service = contrast_to_method(
            labels, absolute, SERVICE_REFERENCE_METHOD
        )
        self.assertEqual(absolute.shape, (len(labels), 36))
        self.assertEqual(len(names), 36)
        actor = labels["candidate_method"].eq("actor").to_numpy()
        p30 = labels["candidate_method"].eq(
            SERVICE_REFERENCE_METHOD
        ).to_numpy()
        self.assertTrue(np.array_equal(
            absolute[actor], np.zeros_like(absolute[actor])
        ))
        self.assertTrue(np.array_equal(
            service[p30], np.zeros_like(service[p30])
        ))

    def test_three_heads_preserve_structural_prediction_zeros(self):
        labels = v31_contexts(count=6)
        signed = labels["candidate_offset_s"].to_numpy(dtype=np.float64) / 30.0
        labels["episode_service_cost_restricted_delta_vs_actor"] = (
            -0.5 * signed + 0.4 * np.square(signed)
        )
        labels["episode_headway_cv_delta_vs_actor"] = 0.1 * signed
        labels["episode_passenger_unserved_rate_delta_vs_actor"] = (
            0.01 * np.maximum(signed, 0.0)
        )
        pipeline, absolute, service, _ = fit_pipeline(
            labels,
            STATE_COLS,
            np.ones(len(labels), dtype=bool),
            rank=2,
            alpha=1e-9,
        )
        predicted = predict_pipeline(pipeline, absolute, service)
        actor = labels["candidate_method"].eq("actor").to_numpy()
        p30 = labels["candidate_method"].eq(
            SERVICE_REFERENCE_METHOD
        ).to_numpy()
        self.assertTrue(np.array_equal(
            predicted["service_vs_p30"][p30], np.zeros(p30.sum())
        ))
        self.assertTrue(np.array_equal(
            predicted["headway_vs_actor"][actor], np.zeros(actor.sum())
        ))
        self.assertTrue(np.array_equal(
            predicted["unserved_vs_actor"][actor], np.zeros(actor.sum())
        ))


class V31SelectorTest(unittest.TestCase):
    def setUp(self):
        self.labels = v31_contexts(count=1)
        self.zero_risk = {
            method: 0.0
            for method in self.labels["candidate_method"].astype(str)
        }

    def test_selector_rejects_best_service_action_when_risky(self):
        service = {
            "actor": 0.1,
            "actor_firstknot_m30": -1.0,
            "actor_firstknot_m15": 0.2,
            "actor_firstknot_p15": -0.5,
            SERVICE_REFERENCE_METHOD: 0.0,
        }
        headway = dict(self.zero_risk)
        headway["actor_firstknot_m30"] = 0.1
        chosen = select_constrained_rows(
            self.labels,
            predictions(
                self.labels,
                service=service,
                headway=headway,
                unserved=self.zero_risk,
            ),
            service_margin=0.0,
        )
        self.assertEqual(
            chosen.loc[0, "candidate_method"], "actor_firstknot_p15"
        )

    def test_selector_defaults_to_p30_without_safe_improvement(self):
        service = {
            method: 0.2
            for method in self.labels["candidate_method"].astype(str)
        }
        service[SERVICE_REFERENCE_METHOD] = 0.0
        chosen = select_constrained_rows(
            self.labels,
            predictions(
                self.labels,
                service=service,
                headway=self.zero_risk,
                unserved=self.zero_risk,
            ),
            service_margin=0.0,
        )
        self.assertEqual(
            chosen.loc[0, "candidate_method"], SERVICE_REFERENCE_METHOD
        )

    def test_selector_uses_actor_when_p30_is_predicted_unsafe(self):
        service = {
            method: 0.2
            for method in self.labels["candidate_method"].astype(str)
        }
        service[SERVICE_REFERENCE_METHOD] = 0.0
        headway = {
            method: 0.1
            for method in self.labels["candidate_method"].astype(str)
        }
        headway["actor"] = 0.0
        unserved = dict(headway)
        chosen = select_constrained_rows(
            self.labels,
            predictions(
                self.labels,
                service=service,
                headway=headway,
                unserved=unserved,
            ),
            service_margin=0.0,
        )
        self.assertEqual(chosen.loc[0, "candidate_method"], "actor")


if __name__ == "__main__":
    unittest.main()
