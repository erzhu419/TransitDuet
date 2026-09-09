"""Focused tests for the frozen V30 expanded-observation branch."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_freqduet_prefix_counterfactual import _validate_job
from scripts.aggregate_freqduet_v30_expanded_prefix import DISCOVERY_CONTRACT
from scripts.audit_freqduet_prefix_counterfactual import build_parser
from scripts.audit_protocol_v6_v28_prefix_common import (
    DECISION_INDICES as V28_DECISION_INDICES,
    EVAL_SEEDS as V28_EVAL_SEEDS,
)
from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CONFIRMATION_DECISION_INDICES,
    CONFIRMATION_POLICY_SEEDS,
    CONFIRMATION_SCENARIO_SEEDS,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_EVAL_EPISODE,
    DISCOVERY_REPLAY_SEED,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPANDED_CONTEXT_COLUMNS,
    LABEL_PROTOCOL_VERSION,
    expected_discovery_jobs,
)
from scripts.fit_freqduet_prefix_value_model import select_rows
from scripts.fit_freqduet_v30_expanded_quadratic_value_model import (
    MODEL_METHODS,
    fit_context_scaler,
    fit_pipeline,
    predict_pipeline,
    transform_features,
)
from scripts.submit_freqduet_v30_expanded_prefix_scheduleurm import (
    DEFAULT_NODES,
    build_specs,
)
from tests.test_protocol_v6_v28_prefix_matrix import (
    aggregated_labels_for_job,
    labels_for_job,
    meta_for_job,
)


COMMIT = "1" * 40


def v30_job() -> tuple[dict, pd.DataFrame]:
    meta = meta_for_job()
    meta.update({
        "protocol_version": LABEL_PROTOCOL_VERSION,
        "scenario_seed": DISCOVERY_SCENARIO_SEEDS[0],
        "decision_index": DISCOVERY_DECISION_INDICES[0],
        "eval_episode": DISCOVERY_EVAL_EPISODE,
        "replay_seed": DISCOVERY_REPLAY_SEED,
    })
    meta["target_identity"]["decision_index"] = DISCOVERY_DECISION_INDICES[0]
    labels = labels_for_job()
    labels["scenario_seed"] = DISCOVERY_SCENARIO_SEEDS[0]
    labels["dispatch_index"] = DISCOVERY_DECISION_INDICES[0]
    labels["eval_episode"] = DISCOVERY_EVAL_EPISODE
    labels["waiting_total_pre"] = 125.0
    labels["headway_cv_active_pre"] = 0.2
    return meta, labels


def prepared_contexts(count: int = 4) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for index in range(count):
        labels = aggregated_labels_for_job()
        labels = labels[labels["candidate_method"].isin(MODEL_METHODS)].copy()
        labels["decision_index"] = index + 1
        labels["dispatch_index"] = index + 1
        labels["waiting_total_pre"] = float(50 + 25 * index)
        labels["headway_cv_active_pre"] = float(0.05 + 0.03 * index)
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


class V30RosterAndContractTest(unittest.TestCase):
    def test_discovery_and_confirmation_rosters_are_frozen_and_disjoint(self):
        jobs = expected_discovery_jobs()
        self.assertEqual(len(jobs), 448)
        self.assertEqual(len(set(jobs)), 448)
        self.assertTrue(set(DISCOVERY_SCENARIO_SEEDS).isdisjoint(V28_EVAL_SEEDS))
        self.assertTrue(set(DISCOVERY_DECISION_INDICES).isdisjoint(
            V28_DECISION_INDICES
        ))
        self.assertTrue(set(CONFIRMATION_POLICY_SEEDS).isdisjoint(
            DISCOVERY_TRAIN_SEEDS
        ))
        self.assertTrue(set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            DISCOVERY_SCENARIO_SEEDS
        ))
        self.assertTrue(set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            DISCOVERY_DECISION_INDICES
        ))
        self.assertTrue(set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            V28_DECISION_INDICES
        ))

    def test_custom_protocol_version_is_written_by_shared_auditor(self):
        args = build_parser().parse_args([
            "--protocol-version", LABEL_PROTOCOL_VERSION,
            "--config", "config",
            "--train-seed", "1",
            "--checkpoint-dir", "/tmp/checkpoint",
            "--checkpoint-ep", "39",
            "--scenario-seed", "2",
            "--decision-index", "3",
            "--out-dir", "/tmp/output",
        ])
        self.assertEqual(args.protocol_version, LABEL_PROTOCOL_VERSION)

    def test_v30_contract_accepts_required_causal_context(self):
        meta, labels = v30_job()
        key, validated, commit = _validate_job(
            Path("/tmp/prefix_counterfactual_meta.json"),
            meta,
            labels,
            expected_commit=COMMIT,
            contract=DISCOVERY_CONTRACT,
        )
        self.assertEqual(key, (
            DISCOVERY_TRAIN_SEEDS[0],
            DISCOVERY_SCENARIO_SEEDS[0],
            DISCOVERY_DECISION_INDICES[0],
        ))
        self.assertEqual(commit, COMMIT)
        self.assertEqual(len(validated), 6)

    def test_v30_contract_rejects_missing_expanded_context(self):
        meta, labels = v30_job()
        labels = labels.drop(columns=[EXPANDED_CONTEXT_COLUMNS[0]])
        with self.assertRaisesRegex(RuntimeError, "label columns missing"):
            _validate_job(
                Path("/tmp/prefix_counterfactual_meta.json"),
                meta,
                labels,
                expected_commit=COMMIT,
                contract=DISCOVERY_CONTRACT,
            )

    def test_v30_contract_rejects_post_branch_context_changes(self):
        meta, labels = v30_job()
        labels.loc[labels.index[-1], "waiting_total_pre"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "causal context changed"):
            _validate_job(
                Path("/tmp/prefix_counterfactual_meta.json"),
                meta,
                labels,
                expected_commit=COMMIT,
                contract=DISCOVERY_CONTRACT,
            )

    def test_bulk_specs_pin_every_job_to_hpc_nodes(self):
        specs = build_specs(
            commit="a" * 40,
            run_name="v30-test",
            nodes=DEFAULT_NODES,
            ram_mb=1536,
            priority="high",
            allow_duplicate=False,
        )
        self.assertEqual(len(specs), 448)
        self.assertEqual(len({spec["signature"] for spec in specs}), 448)
        counts = Counter(spec["require_node"] for spec in specs)
        self.assertEqual(set(counts), set(DEFAULT_NODES))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(
            LABEL_PROTOCOL_VERSION in str(spec["cmd"]) for spec in specs
        ))
        self.assertTrue(all(
            str(spec["require_node"]).startswith("node00") for spec in specs
        ))


class V30ModelTest(unittest.TestCase):
    def test_expanded_quadratic_design_is_12d_and_zero_at_actor(self):
        labels = prepared_contexts()
        scaler = fit_context_scaler(
            labels, np.ones(len(labels), dtype=bool)
        )
        design, names = transform_features(labels, scaler)
        self.assertEqual(design.shape, (len(labels), 12))
        self.assertEqual(len(names), 12)
        self.assertEqual(
            scaler["context_columns"], EXPANDED_CONTEXT_COLUMNS
        )
        actor = labels["candidate_method"].eq("actor").to_numpy()
        self.assertTrue(np.array_equal(
            design[actor], np.zeros_like(design[actor])
        ))

    def test_scaler_excludes_heldout_context(self):
        labels = prepared_contexts(count=3)
        train_mask = labels["decision_index"].le(2).to_numpy()
        scaler = fit_context_scaler(labels, train_mask)
        np.testing.assert_allclose(scaler["context_mean"], [62.5, 0.065])
        self.assertEqual(scaler["fitted_actor_contexts"], 2)

    def test_quadratic_pipeline_can_select_interior_action(self):
        labels = prepared_contexts(count=2)
        signed = labels["candidate_offset_s"].to_numpy(dtype=np.float64) / 30.0
        positive = np.maximum(signed, 0.0)
        negative = np.maximum(-signed, 0.0)
        labels["episode_service_cost_restricted_delta_vs_actor"] = (
            -positive + 0.75 * np.square(positive)
            + 0.5 * negative + 0.5 * np.square(negative)
        )
        pipeline, design, _ = fit_pipeline(
            labels, np.ones(len(labels), dtype=bool), alpha=1e-9
        )
        first_mask = labels["decision_index"].eq(1).to_numpy()
        selected = select_rows(
            labels.loc[first_mask].reset_index(drop=True),
            predict_pipeline(pipeline, design[first_mask]),
            guard_margin=0.0,
        )
        self.assertEqual(
            selected.loc[0, "candidate_method"], "actor_firstknot_p15"
        )


if __name__ == "__main__":
    unittest.main()
