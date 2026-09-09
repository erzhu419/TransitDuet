"""Focused tests for the V28 exact-prefix matrix and deployable fitter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_freqduet_prefix_counterfactual import _validate_job
from scripts.audit_protocol_v6_v28_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    DECISION_INDICES,
    EVAL_EPISODE,
    EVAL_SEEDS,
    EXPECTED_METHODS,
    OFFSETS_S,
    OUTCOME_DELTAS,
    PROTOCOL_VERSION,
    REPLAY_SEED,
    TRAIN_SEEDS,
    checkpoint_dir,
    expected_jobs,
)
from scripts.fit_freqduet_prefix_value_model import build_features, select_rows


COMMIT = "1" * 40


def labels_for_job() -> pd.DataFrame:
    rows = []
    offsets = [0.0, -30.0, -15.0, 0.0, 15.0, 30.0]
    for method, offset in zip(EXPECTED_METHODS, offsets):
        identity = method in {"actor", "actor_firstknot_0"}
        row = {
            "train_seed": TRAIN_SEEDS[0],
            "scenario_seed": EVAL_SEEDS[0],
            "decision_index": DECISION_INDICES[0],
            "eval_episode": EVAL_EPISODE,
            "candidate_method": method,
            "candidate_offset_s": offset,
            "candidate_action_linf_delta_s": 0.0 if identity else abs(offset),
            "actor_action_json": json.dumps([1.0, 2.0]),
            "candidate_action_json": json.dumps(
                [1.0 if identity else 1.0 + offset, 2.0]
            ),
            "upper_state_dim": 2,
            "upper_state_000": 0.25,
            "upper_state_001": -0.5,
            "snapshot_time_s": 22000.0,
        }
        for column in OUTCOME_DELTAS.values():
            row[column] = 0.0 if identity else offset / 1000.0
        rows.append(row)
    return pd.DataFrame(rows)


def meta_for_job() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "mechanical_pass",
        "effect_evidence": False,
        "source": {"commit": COMMIT, "tracked_dirty": False},
        "config": f"/tmp/{CONFIG}.yaml",
        "train_seed": TRAIN_SEEDS[0],
        "checkpoint_dir": str(checkpoint_dir(TRAIN_SEEDS[0])),
        "checkpoint_ep": CHECKPOINT_EP,
        "policy_digest": "policy-one",
        "eval_episode": EVAL_EPISODE,
        "scenario_seed": EVAL_SEEDS[0],
        "replay_seed": REPLAY_SEED,
        "decision_index": DECISION_INDICES[0],
        "target_identity": {
            "decision_index": DECISION_INDICES[0],
            "write_terminal_dispatch": True,
        },
        "offsets_s": OFFSETS_S,
        "candidate_parameterization": "same_direction_first_bernstein_knot_v1",
        "terminal_dispatch_preserved": True,
        "checks": {
            "actor_repeat_prefix_exact": True,
            "actor_repeat_episode_exact": True,
            "candidate_prefixes_exact": True,
            "zero_offset_episode_exact": True,
            "policy_checkpoint_exact": True,
            "global_and_isolated_rng_exact": True,
            "nonzero_action_response": True,
        },
    }


class V28RosterTest(unittest.TestCase):
    def test_frozen_grid_has_128_unique_jobs(self):
        jobs = expected_jobs()
        self.assertEqual(len(jobs), 128)
        self.assertEqual(len(set(jobs)), 128)
        self.assertEqual(DECISION_INDICES[-1], 84)

    def test_registered_job_validates(self):
        labels = labels_for_job()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix_counterfactual_meta.json"
            key, validated, commit = _validate_job(
                path, meta_for_job(), labels, expected_commit=COMMIT
            )
        self.assertEqual(key, (TRAIN_SEEDS[0], EVAL_SEEDS[0], DECISION_INDICES[0]))
        self.assertEqual(commit, COMMIT)
        self.assertEqual(len(validated), len(EXPECTED_METHODS))

    def test_invalid_identity_delta_fails_closed(self):
        labels = labels_for_job()
        labels.loc[
            labels["candidate_method"] == "actor_firstknot_0",
            next(iter(OUTCOME_DELTAS.values())),
        ] = 1e-4
        with self.assertRaisesRegex(RuntimeError, "identity branch changed"):
            _validate_job(
                Path("/tmp/prefix_counterfactual_meta.json"),
                meta_for_job(),
                labels,
                expected_commit=COMMIT,
            )


class V28ModelTest(unittest.TestCase):
    def test_features_use_only_state_and_actions(self):
        labels = labels_for_job()
        labels["actor_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["actor_action_json"]
        ]
        labels["candidate_action"] = [
            np.asarray(json.loads(value), dtype=np.float64)
            for value in labels["candidate_action_json"]
        ]
        design, names = build_features(
            labels, ["upper_state_000", "upper_state_001"]
        )
        self.assertEqual(design.shape, (len(labels), len(names)))
        self.assertTrue(np.isfinite(design).all())
        self.assertFalse(any(
            token in name.lower()
            for name in names
            for token in ("domain", "config", "seed", "future", "outcome")
        ))

    def test_prediction_guard_keeps_actor_without_margin(self):
        labels = labels_for_job()
        predictions = np.asarray([0.0, -0.01, -0.005, 0.0, 0.002, 0.003])
        chosen = select_rows(labels, predictions, guard_margin=0.02)
        self.assertEqual(chosen.loc[0, "candidate_method"], "actor")

    def test_prediction_guard_allows_material_candidate(self):
        labels = labels_for_job()
        predictions = np.asarray([0.0, -0.03, -0.005, 0.0, 0.002, 0.003])
        chosen = select_rows(labels, predictions, guard_margin=0.02)
        self.assertEqual(chosen.loc[0, "candidate_method"], "actor_firstknot_m30")


if __name__ == "__main__":
    unittest.main()
