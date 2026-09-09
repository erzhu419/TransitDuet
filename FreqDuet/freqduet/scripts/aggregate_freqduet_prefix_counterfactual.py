#!/usr/bin/env python3
"""Strictly aggregate the frozen V28 exact-prefix label matrix."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v28_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    DECISION_INDICES,
    EVAL_EPISODE,
    EVAL_SEEDS,
    EXPECTED_METHODS,
    MATRIX_PROTOCOL_VERSION,
    OFFSETS_S,
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
    PROTOCOL_VERSION,
    REPLAY_SEED,
    TRAIN_SEEDS,
    checkpoint_dir,
    expected_jobs,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_KEYS = ["train_seed", "scenario_seed", "decision_index"]
CONTEXT_KEYS = JOB_KEYS + ["eval_episode"]
LABEL_CONTEXT_KEYS = [
    "train_seed", "scenario_seed", "dispatch_index", "eval_episode",
]


def bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan, np.nan
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, data.size, size=(int(n_boot), data.size))
    means = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _source_commit(meta: dict) -> str:
    source = meta.get("source", {}) or {}
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)), f"invalid source commit {commit!r}")
    _require(source.get("tracked_dirty") is False, "source snapshot is tracked-dirty")
    return commit


def _analysis_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid aggregation source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "aggregation source snapshot is tracked-dirty")
    return commit


def _load_job(meta_path: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads(meta_path.read_text())
    labels_path = meta_path.with_name("prefix_counterfactual_labels.csv")
    _require(labels_path.is_file(), f"missing labels next to {meta_path}")
    labels = pd.read_csv(labels_path)
    return meta, labels


def _validate_job(
    meta_path: Path,
    meta: dict,
    labels: pd.DataFrame,
    *,
    expected_commit: str | None,
) -> tuple[tuple[int, int, int], pd.DataFrame, str]:
    _require(meta.get("protocol_version") == PROTOCOL_VERSION,
             f"{meta_path}: wrong protocol")
    _require(meta.get("status") == "mechanical_pass", f"{meta_path}: not a pass")
    _require(meta.get("effect_evidence") is False,
             f"{meta_path}: mechanical job claims effect evidence")
    checks = meta.get("checks", {}) or {}
    _require(bool(checks) and all(value is True for value in checks.values()),
             f"{meta_path}: incomplete mechanical checks")
    commit = _source_commit(meta)
    if expected_commit:
        _require(commit == expected_commit.lower(),
                 f"{meta_path}: source {commit} != {expected_commit}")

    train_seed = int(meta.get("train_seed"))
    eval_seed = int(meta.get("scenario_seed"))
    decision_index = int(meta.get("decision_index"))
    key = (train_seed, eval_seed, decision_index)
    _require(train_seed in TRAIN_SEEDS, f"{meta_path}: unregistered train seed")
    _require(eval_seed in EVAL_SEEDS, f"{meta_path}: unregistered eval seed")
    _require(decision_index in DECISION_INDICES,
             f"{meta_path}: unregistered decision index")
    _require(int(meta.get("checkpoint_ep")) == CHECKPOINT_EP,
             f"{meta_path}: wrong checkpoint episode")
    _require(int(meta.get("eval_episode")) == EVAL_EPISODE,
             f"{meta_path}: wrong evaluation episode")
    _require(int(meta.get("replay_seed")) == REPLAY_SEED,
             f"{meta_path}: wrong replay seed")
    _require([float(x) for x in meta.get("offsets_s", [])] == OFFSETS_S,
             f"{meta_path}: wrong candidate offsets")
    _require(Path(str(meta.get("config", ""))).stem == CONFIG,
             f"{meta_path}: wrong config")
    _require(Path(str(meta.get("checkpoint_dir", ""))) == checkpoint_dir(train_seed),
             f"{meta_path}: wrong checkpoint directory")
    _require(meta.get("candidate_parameterization") ==
             "same_direction_first_bernstein_knot_v1",
             f"{meta_path}: wrong candidate parameterization")
    _require(meta.get("terminal_dispatch_preserved") is True,
             f"{meta_path}: executable terminal dispatch was not retained")
    target = meta.get("target_identity", {}) or {}
    _require(int(target.get("decision_index")) == decision_index,
             f"{meta_path}: target identity mismatch")
    _require(target.get("write_terminal_dispatch") is True,
             f"{meta_path}: target is not executable terminal dispatch")

    required_columns = set(LABEL_CONTEXT_KEYS + [
        "candidate_method", "candidate_offset_s", "candidate_action_linf_delta_s",
        "actor_action_json", "candidate_action_json", *OUTCOME_DELTAS.values(),
    ])
    missing = sorted(required_columns - set(labels.columns))
    _require(not missing, f"{meta_path}: label columns missing {missing}")
    _require(len(labels) == len(EXPECTED_METHODS),
             f"{meta_path}: expected {len(EXPECTED_METHODS)} rows, got {len(labels)}")
    _require(labels["candidate_method"].astype(str).tolist() == EXPECTED_METHODS,
             f"{meta_path}: candidate roster/order mismatch")
    _require(not labels["candidate_method"].duplicated().any(),
             f"{meta_path}: duplicate candidate methods")
    for column, expected in (
        ("train_seed", train_seed),
        ("scenario_seed", eval_seed),
        ("dispatch_index", decision_index),
        ("eval_episode", EVAL_EPISODE),
    ):
        values = pd.to_numeric(labels[column], errors="coerce")
        _require(values.notna().all() and values.eq(expected).all(),
                 f"{meta_path}: {column} mismatch")
    for column in OUTCOME_DELTAS.values():
        values = pd.to_numeric(labels[column], errors="coerce")
        _require(np.isfinite(values.to_numpy(dtype=np.float64)).all(),
                 f"{meta_path}: non-finite {column}")
    identity = labels[labels["candidate_method"].isin(
        ["actor", "actor_firstknot_0"])]
    for column in OUTCOME_DELTAS.values():
        _require(np.array_equal(
            pd.to_numeric(identity[column], errors="coerce").to_numpy(dtype=np.float64),
            np.zeros(2, dtype=np.float64)),
            f"{meta_path}: identity branch changed {column}")
    nonzero = labels[~labels["candidate_method"].isin(
        ["actor", "actor_firstknot_0"])]
    response = pd.to_numeric(
        nonzero["candidate_action_linf_delta_s"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    _require(np.isfinite(response).all() and np.any(response > 0.0),
             f"{meta_path}: no nonzero action response")

    out = labels.copy()
    out["decision_index"] = decision_index
    out["source_commit"] = commit
    out["policy_digest"] = str(meta.get("policy_digest", ""))
    out["job_dir"] = str(meta_path.parent)
    return key, out, commit


def _block_summary(frame: pd.DataFrame, metric: str, *, seed: int) -> dict[str, float]:
    blocks = (
        frame.groupby(["train_seed", "scenario_seed"], sort=True)[metric]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    low, high = bootstrap_ci(blocks, seed=seed)
    return {
        "mean": float(np.mean(blocks)),
        "ci_low": low,
        "ci_high": high,
        "blocks": int(blocks.size),
    }


def _candidate_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in EXPECTED_METHODS:
        group = labels[labels["candidate_method"] == method]
        row: dict[str, object] = {
            "candidate_method": method,
            "candidate_offset_s": float(pd.to_numeric(
                group["candidate_offset_s"], errors="raise").iloc[0]),
            "contexts": int(len(group)),
        }
        for index, (name, column) in enumerate(OUTCOME_DELTAS.items()):
            stats = _block_summary(group, column, seed=28028 + index)
            for stat, value in stats.items():
                row[f"{name}_{stat}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _oracle_summary(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    eligible = labels[labels["candidate_method"] != "actor_firstknot_0"].copy()
    eligible[PRIMARY_DELTA] = pd.to_numeric(eligible[PRIMARY_DELTA], errors="raise")
    eligible["_tie_priority"] = np.where(
        eligible["candidate_method"].eq("actor"), 0, 1
    )
    ordered = eligible.sort_values(
        CONTEXT_KEYS + [PRIMARY_DELTA, "_tie_priority", "candidate_offset_s"],
        kind="mergesort",
    )
    oracle = ordered.groupby(CONTEXT_KEYS, sort=True, as_index=False).first()
    oracle = oracle.drop(columns=["_tie_priority"])
    outcome = {
        name: _block_summary(oracle, column, seed=28128 + index)
        for index, (name, column) in enumerate(OUTCOME_DELTAS.items())
    }
    summary: dict[str, object] = {
        "effect_evidence": "exploratory_oracle_only",
        "selection_unit": "one exact-prefix upper decision",
        "contexts": int(len(oracle)),
        "non_actor_fraction": float(
            (~oracle["candidate_method"].eq("actor")).mean()),
        "selected_method_counts": {
            str(key): int(value)
            for key, value in oracle["candidate_method"].value_counts().sort_index().items()
        },
        "outcomes": outcome,
    }
    return oracle, summary


def aggregate(
    jobs_root: Path,
    out_dir: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, object]:
    jobs_root = Path(jobs_root).resolve()
    out_dir = Path(out_dir).resolve()
    analysis_commit = _analysis_commit()
    meta_files = sorted(jobs_root.rglob("prefix_counterfactual_meta.json"))
    expected = set(expected_jobs())
    _require(len(meta_files) == len(expected),
             f"expected {len(expected)} job metadata files, found {len(meta_files)}")

    observed: dict[tuple[int, int, int], Path] = {}
    parts: list[pd.DataFrame] = []
    commits: set[str] = set()
    for meta_path in meta_files:
        meta, labels = _load_job(meta_path)
        key, validated, commit = _validate_job(
            meta_path, meta, labels, expected_commit=expected_commit
        )
        _require(key not in observed,
                 f"duplicate matrix job {key}: {observed.get(key)} and {meta_path}")
        observed[key] = meta_path
        parts.append(validated)
        commits.add(commit)
    _require(set(observed) == expected,
             f"matrix roster mismatch missing={sorted(expected - set(observed))} "
             f"extra={sorted(set(observed) - expected)}")
    _require(len(commits) == 1, f"matrix mixes source commits: {sorted(commits)}")

    labels = pd.concat(parts, ignore_index=True)
    expected_rows = len(expected) * len(EXPECTED_METHODS)
    _require(len(labels) == expected_rows,
             f"expected {expected_rows} rows, found {len(labels)}")
    policy_counts = labels.groupby("train_seed")["policy_digest"].nunique()
    _require(policy_counts.eq(1).all(),
             f"policy checkpoint changed within train seed: {policy_counts.to_dict()}")
    _require(labels.groupby("policy_digest")["train_seed"].nunique().eq(1).all(),
             "one policy digest is assigned to multiple train seeds")

    candidate = _candidate_summary(labels)
    oracle, oracle_summary = _oracle_summary(labels)
    target_times = (
        labels[labels["candidate_method"] == "actor"]
        .groupby("decision_index")["snapshot_time_s"]
        .agg(["min", "mean", "max"])
        .reset_index()
    )
    _require(target_times["mean"].is_monotonic_increasing,
             "registered decisions are not temporally ordered")

    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "prefix_counterfactual_all.csv", index=False)
    candidate.to_csv(out_dir / "candidate_block_summary.csv", index=False)
    oracle.to_csv(out_dir / "oracle_selected_contexts.csv", index=False)
    target_times.to_csv(out_dir / "decision_time_audit.csv", index=False)
    (out_dir / "oracle_summary.json").write_text(
        json.dumps(oracle_summary, indent=2, sort_keys=True) + "\n"
    )
    manifest: dict[str, object] = {
        "protocol_version": MATRIX_PROTOCOL_VERSION,
        "label_protocol_version": PROTOCOL_VERSION,
        "status": "strict_complete",
        "strict_complete": True,
        "effect_evidence": "exploratory_labels_only",
        "source_commit": next(iter(commits)),
        "rollout_source_commit": next(iter(commits)),
        "aggregation_source_commit": analysis_commit,
        "config": CONFIG,
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "decision_indices": DECISION_INDICES,
        "offsets_s": OFFSETS_S,
        "checkpoint_ep": CHECKPOINT_EP,
        "eval_episode": EVAL_EPISODE,
        "replay_seed": REPLAY_SEED,
        "jobs": len(observed),
        "rows": len(labels),
        "contexts": len(expected),
        "policy_digests_by_train_seed": {
            str(seed): str(labels.loc[
                labels["train_seed"].eq(seed), "policy_digest"
            ].iloc[0])
            for seed in TRAIN_SEEDS
        },
        "checks": {
            "exact_cartesian_roster": True,
            "all_mechanical_gates_pass": True,
            "one_policy_per_train_seed": True,
            "identity_candidates_exact": True,
            "all_outcomes_finite": True,
            "executable_terminal_dispatch_retained": True,
            "decision_times_monotone": True,
        },
    }
    (out_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args()
    try:
        manifest = aggregate(
            args.jobs_root,
            args.out_dir,
            expected_commit=args.expected_source_commit,
        )
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "aggregate_invalid.json").write_text(json.dumps({
            "protocol_version": MATRIX_PROTOCOL_VERSION,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE V28 aggregate status={manifest['status']} jobs={manifest['jobs']} "
        f"rows={manifest['rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
