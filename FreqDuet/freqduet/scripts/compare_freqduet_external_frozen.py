#!/usr/bin/env python3
"""Paired learned-policy comparisons against frozen external baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from scripts.analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        validate_csv_artifact,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        validate_csv_artifact,
    )


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_freqduet_protocol_v2_matrix import (
    METRIC_DIRECTIONS,
    hierarchical_bootstrap,
    holm_adjusted_pvalues,
    paired_sign_flip_p,
)
from scripts.run_freqduet_external_baselines import (
    BASELINE_VARIANTS,
    V6_EXECUTION_METRICS,
    V6_PER_SEED_PRIMARY_KEY,
    V6_PROTOCOL_VERSION,
    V6_SUMMARY_MANIFEST_VERSION,
    external_evaluator_fingerprint,
    git_provenance,
    scenario_contract_fingerprint,
    source_fingerprint,
)
from scripts import run_freqduet_protocol_v2_matrix as matrix_protocol


METRICS = [
    "service_cost_restricted",
    "service_cost_observed",
    "restricted_wait_horizon_min",
    "avg_wait_observed_min",
    "restricted_in_vehicle_horizon_min",
    "avg_in_vehicle_observed_min",
    "restricted_total_journey_horizon_min",
    "avg_total_journey_observed_min",
    "holding_vehicle_seconds",
    "holding_passenger_seconds",
    "fleet_denied_dispatch_events",
    "fleet_denied_trips",
    "fleet_readiness_delay_mean_s",
    "fleet_readiness_delay_max_s",
    "passenger_unserved_rate",
    "headway_cv",
    "fleet_overshoot",
    "trip_launch_rate",
    "trip_completion_rate",
]
V5_METRICS = METRICS + [
    "holding_vehicle_seconds_per_launched_trip",
    "holding_passenger_min_per_generated",
    "fleet_denied_trip_rate",
]
V6_METRICS = list(dict.fromkeys(V5_METRICS + V6_EXECUTION_METRICS))
V6_LEARNED_PRIMARY_KEY = ["config", "train_seed", "eval_seed"]
V6_PAIR_PRIMARY_KEY = [
    "learned_config", "baseline_config", "baseline_method",
    "train_seed", "eval_seed",
]
V6_COMPARISON_SUMMARY_PRIMARY_KEY = [
    "learned_config", "baseline_config", "baseline_method",
]
V6_COMPARISON_MANIFEST_VERSION = "freqduet-external-comparison-v6"
REQUIRED_EXTERNAL_METHODS = tuple(BASELINE_VARIANTS)


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _validate_tapes(
    learned: pd.DataFrame,
    baseline: pd.DataFrame,
    method: str,
) -> None:
    learned_tapes = learned.groupby("eval_seed")["scenario_tape_id"].nunique()
    if not learned_tapes.eq(1).all():
        raise ValueError("learned policies disagree on scenario tape IDs")
    baseline_tapes = baseline.groupby("eval_seed")["scenario_tape_id"].nunique()
    if not baseline_tapes.eq(1).all():
        raise ValueError(f"{method}: duplicate baseline tape IDs")
    learned_map = learned.groupby("eval_seed")["scenario_tape_id"].first()
    baseline_map = baseline.groupby("eval_seed")["scenario_tape_id"].first()
    if set(learned_map.index) != set(baseline_map.index):
        raise ValueError(f"{method}: evaluation seed set does not match learned")
    mismatch = learned_map.astype(str) != baseline_map.astype(str)
    if mismatch.any():
        raise ValueError(
            f"{method}: common-random-number tape mismatch at "
            f"{learned_map.index[mismatch].tolist()}")


def _validate_fingerprint_record(record: object, label: str) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"missing {label} fingerprint")
    sha = str(record.get("sha256", ""))
    if len(sha) != 64:
        raise ValueError(f"invalid {label} fingerprint")
    if "payload" in record:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"invalid {label} fingerprint payload")
        if canonical_json_sha256(payload) != sha:
            raise ValueError(f"{label} fingerprint payload does not match hash")
    return record


def _scenario_contract_for(
    manifest: dict,
    config: str,
    label: str,
) -> dict:
    contracts = manifest.get("scenario_contract_fingerprints")
    if isinstance(contracts, dict) and config in contracts:
        record = contracts[config]
    else:
        record = (
            manifest.get("scenario_contract")
            or manifest.get("scenario_contract_fingerprint"))
    return _validate_fingerprint_record(record, f"{label} scenario contract")


def _manifest_artifact(
    manifest: dict,
    path: Path,
    primary_key: list[str],
    label: str,
    artifact_key: str,
) -> pd.DataFrame:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{label} manifest has no artifact records")
    record = artifacts.get(artifact_key)
    if record is None:
        raise ValueError(
            f"{label} manifest does not bind artifact {artifact_key}")
    return validate_csv_artifact(
        path, record, expected_primary_key=primary_key)


def _load_and_validate_manifests(
    learned_path: Path,
    baseline_path: Path,
    learned_manifest_path: Path | None,
    baseline_manifest_path: Path | None,
) -> tuple[dict[str, object], dict, dict]:
    learned_manifest_path = (
        learned_manifest_path or learned_path.with_name("matrix_manifest.json"))
    baseline_manifest_path = (
        baseline_manifest_path
        or baseline_path.with_name("external_baselines_summary.json"))
    if not learned_manifest_path.exists():
        raise ValueError(f"missing learned matrix manifest {learned_manifest_path}")
    if not baseline_manifest_path.exists():
        raise ValueError(
            f"missing external baseline manifest {baseline_manifest_path}")
    learned = json.loads(learned_manifest_path.read_text())
    baseline = json.loads(baseline_manifest_path.read_text())
    protocol_version = str(learned.get("protocol_version", ""))
    if protocol_version not in {
            "freqduet-eval-v4", "freqduet-eval-v5",
            V6_PROTOCOL_VERSION}:
        raise ValueError(
            f"unsupported learned protocol version {protocol_version!r}")
    required_learned = {
        "protocol_version": protocol_version,
        "strict_complete": True,
        "common_random_numbers_verified": True,
        "run_manifests_verified": True,
    }
    for key, expected in required_learned.items():
        if learned.get(key) != expected:
            raise ValueError(
                f"learned matrix manifest {key}={learned.get(key)!r}, "
                f"expected {expected!r}")
    required_baseline = {
        "direct_scenario_seeds": True,
        "run_manifests_verified": True,
    }
    for key, expected in required_baseline.items():
        if baseline.get(key) != expected:
            raise ValueError(
                f"external manifest {key}={baseline.get(key)!r}, "
                f"expected {expected!r}")
    if protocol_version == V6_PROTOCOL_VERSION:
        expected_matrix_version = str(getattr(
            matrix_protocol,
            "MATRIX_MANIFEST_VERSION",
            "freqduet-matrix-manifest-v2",
        ))
        if learned.get("manifest_version") != expected_matrix_version:
            raise ValueError("learned V6 matrix manifest version mismatch")
        required_v6_baseline = {
            "manifest_version": V6_SUMMARY_MANIFEST_VERSION,
            "protocol_version": V6_PROTOCOL_VERSION,
            "strict_complete": True,
            "protocol_versions": [V6_PROTOCOL_VERSION],
        }
        for key, expected in required_v6_baseline.items():
            if baseline.get(key) != expected:
                raise ValueError(
                    f"external V6 manifest {key}={baseline.get(key)!r}, "
                    f"expected {expected!r}")
        if not isinstance(learned.get("artifacts"), dict):
            raise ValueError("learned V6 matrix manifest has no artifact records")
        if not isinstance(baseline.get("artifacts"), dict):
            raise ValueError("external V6 manifest has no artifact records")
    learned_source = learned.get("run_source_fingerprint") or {}
    learned_sha = str(learned_source.get("sha256", ""))
    baseline_sha = str(baseline.get("core_source_sha256", ""))
    evaluator_sha = str(baseline.get("evaluator_source_sha256", ""))
    if len(learned_sha) != 64 or len(baseline_sha) != 64:
        raise ValueError("comparison manifests contain invalid core source hashes")
    if learned_sha != baseline_sha:
        raise ValueError(
            "learned and external baseline core source fingerprints differ")
    if len(evaluator_sha) != 64:
        raise ValueError("external evaluator fingerprint is missing or invalid")
    if protocol_version == V6_PROTOCOL_VERSION:
        baseline_source = _validate_fingerprint_record(
            baseline.get("core_source_fingerprint"), "external core source")
        if str(baseline_source.get("sha256")) != baseline_sha:
            raise ValueError("external core source records disagree")
        evaluator_source = _validate_fingerprint_record(
            baseline.get("evaluator_source_fingerprint"),
            "external evaluator source",
        )
        if str(evaluator_source.get("sha256")) != evaluator_sha:
            raise ValueError("external evaluator source records disagree")
        if str(source_fingerprint().get("sha256")) != learned_sha:
            raise ValueError(
                "current model source does not match the V6 evidence source")
        if str(external_evaluator_fingerprint().get("sha256")) != evaluator_sha:
            raise ValueError(
                "current external evaluator does not match V6 evidence")
        learned_run_git = learned.get("run_git_provenance")
        baseline_run_git = baseline.get("run_git_provenance")
        learned_git = learned.get("git")
        baseline_git = baseline.get("git")
        git_records = {
            label: record
            for label, record in [
                ("learned run", learned_run_git),
                ("external run", baseline_run_git),
                ("learned aggregation", learned_git),
                ("external aggregation", baseline_git),
            ]
        }
        commits = set()
        dirty_values = set()
        for label, record in git_records.items():
            if not isinstance(record, dict):
                raise ValueError(f"missing {label} Git provenance")
            commit = str(record.get("commit", ""))
            dirty = record.get("tracked_dirty")
            if (len(commit) != 40
                    or any(char not in "0123456789abcdef" for char in commit)
                    or not isinstance(dirty, bool)):
                raise ValueError(f"invalid {label} Git provenance")
            commits.add(commit)
            dirty_values.add(dirty)
        if len(commits) != 1 or len(dirty_values) != 1:
            raise ValueError(
                "learned and external V6 Git provenance differs")
        current_git = git_provenance()
        if str(current_git.get("commit", "")) != next(iter(commits)):
            raise ValueError(
                "current comparison Git commit differs from V6 evidence")
    provenance = {
        "learned_manifest": str(learned_manifest_path),
        "baseline_manifest": str(baseline_manifest_path),
        "core_source_sha256": learned_sha,
        "external_evaluator_sha256": evaluator_sha,
        "protocol_version": protocol_version,
    }
    if protocol_version == V6_PROTOCOL_VERSION:
        provenance["git"] = {
            "commit": next(iter(commits)),
            "tracked_dirty": next(iter(dirty_values)),
        }
    return provenance, learned, baseline


def compare(
    learned_path: Path,
    baseline_path: Path,
    out_dir: Path,
    learned_config: str,
    baseline_config: str | None = None,
    methods: list[str] | None = None,
    learned_manifest_path: Path | None = None,
    baseline_manifest_path: Path | None = None,
) -> None:
    provenance, learned_manifest, baseline_manifest = (
        _load_and_validate_manifests(
        learned_path,
        baseline_path,
        learned_manifest_path,
        baseline_manifest_path,
    ))
    protocol_version = str(provenance["protocol_version"])
    if protocol_version == V6_PROTOCOL_VERSION:
        learned_all = _manifest_artifact(
            learned_manifest,
            learned_path,
            V6_LEARNED_PRIMARY_KEY,
            "learned V6 matrix",
            "frozen_per_eval.csv",
        )
        baseline_all = _manifest_artifact(
            baseline_manifest,
            baseline_path,
            V6_PER_SEED_PRIMARY_KEY,
            "external V6 baseline",
            "external_baselines_per_seed.csv",
        )
        metrics = V6_METRICS
    else:
        learned_all = pd.read_csv(learned_path)
        baseline_all = pd.read_csv(baseline_path)
        metrics = (
            V5_METRICS
            if protocol_version == "freqduet-eval-v5" else METRICS)
    required_common = {"config", "eval_seed", "scenario_tape_id", *metrics}
    if protocol_version == V6_PROTOCOL_VERSION:
        required_common.add("scenario_contract_sha256")
    _require_columns(
        learned_all, required_common | {"train_seed"}, learned_path)
    _require_columns(
        baseline_all, required_common | {"method"}, baseline_path)

    learned = learned_all[
        learned_all["config"].astype(str).eq(str(learned_config))
    ].copy()
    baseline_name = str(baseline_config or learned_config)
    baselines = baseline_all[
        baseline_all["config"].astype(str).eq(baseline_name)
    ].copy()
    if learned.empty:
        raise ValueError(f"no learned rows for {learned_config}")
    if baselines.empty:
        raise ValueError(f"no baseline rows for {baseline_name}")
    if learned.duplicated(["train_seed", "eval_seed"]).any():
        raise ValueError("learned matrix has duplicate train/eval pairs")
    if set(learned["protocol_version"].astype(str)) != {protocol_version}:
        raise ValueError("learned input protocol does not match its manifest")
    if ("protocol_version" not in baselines
            or set(baselines["protocol_version"].astype(str))
            != {protocol_version}):
        raise ValueError("external baseline protocol does not match learned")

    if protocol_version == V6_PROTOCOL_VERSION:
        learned_configs = [
            str(value) for value in learned_manifest.get("configs", [])]
        baseline_configs = [
            str(value) for value in baseline_manifest.get("configs", [])]
        learned_train_seeds = [
            int(value) for value in learned_manifest.get("train_seeds", [])]
        learned_eval_seeds = [
            int(value) for value in learned_manifest.get("eval_seeds", [])]
        baseline_eval_seeds = [
            int(value) for value in baseline_manifest.get("eval_seeds", [])]
        baseline_methods = [
            str(value) for value in baseline_manifest.get("methods", [])]
        for label, values in [
            ("learned configs", learned_configs),
            ("baseline configs", baseline_configs),
            ("learned train seeds", learned_train_seeds),
            ("learned eval seeds", learned_eval_seeds),
            ("baseline eval seeds", baseline_eval_seeds),
            ("baseline methods", baseline_methods),
        ]:
            if not values or len(values) != len(set(values)):
                raise ValueError(f"V6 manifest {label} are missing or duplicated")
        if set(learned_all["config"].astype(str)) != set(learned_configs):
            raise ValueError("learned V6 artifact config set mismatches manifest")
        if set(baseline_all["config"].astype(str)) != set(baseline_configs):
            raise ValueError("external V6 artifact config set mismatches manifest")
        if set(learned_all["train_seed"].astype(int)) != set(
                learned_train_seeds):
            raise ValueError("learned V6 train seed set mismatches manifest")
        if set(learned_all["eval_seed"].astype(int)) != set(
                learned_eval_seeds):
            raise ValueError("learned V6 eval seed set mismatches manifest")
        if set(baseline_all["eval_seed"].astype(int)) != set(
                baseline_eval_seeds):
            raise ValueError("external V6 eval seed set mismatches manifest")
        learned_grid = set(zip(
            learned_all["config"].astype(str),
            learned_all["train_seed"].astype(int),
            learned_all["eval_seed"].astype(int),
        ))
        expected_learned_grid = {
            (config, train_seed, eval_seed)
            for config in learned_configs
            for train_seed in learned_train_seeds
            for eval_seed in learned_eval_seeds
        }
        if learned_grid != expected_learned_grid:
            raise ValueError("learned V6 artifact is not a complete manifest grid")
        if learned_eval_seeds != baseline_eval_seeds:
            raise ValueError("learned and external V6 eval seed contracts differ")
        required_methods = set(REQUIRED_EXTERNAL_METHODS)
        if (len(baseline_methods) != len(REQUIRED_EXTERNAL_METHODS)
                or set(baseline_methods) != required_methods):
            raise ValueError(
                "V6 external manifest must contain exactly fixed_headway, "
                "rule_holding, and rule_mpc")
        observed_methods = set(baseline_all["method"].astype(str))
        if observed_methods != required_methods:
            raise ValueError(
                "V6 external artifact must contain exactly fixed_headway, "
                "rule_holding, and rule_mpc")
        baseline_grid = set(zip(
            baseline_all["config"].astype(str),
            baseline_all["method"].astype(str),
            baseline_all["eval_seed"].astype(int),
        ))
        expected_baseline_grid = {
            (config, method, eval_seed)
            for config in baseline_configs
            for method in REQUIRED_EXTERNAL_METHODS
            for eval_seed in baseline_eval_seeds
        }
        if baseline_grid != expected_baseline_grid:
            raise ValueError("external V6 artifact is not a complete manifest grid")
        requested_methods = (
            list(REQUIRED_EXTERNAL_METHODS) if methods is None
            else [str(value) for value in methods])
        if (len(requested_methods) != len(REQUIRED_EXTERNAL_METHODS)
                or set(requested_methods) != required_methods):
            raise ValueError(
                "V6 comparison methods must be exactly fixed_headway, "
                "rule_holding, and rule_mpc")
        selected_methods = list(REQUIRED_EXTERNAL_METHODS)
        learned_contract = _scenario_contract_for(
            learned_manifest, str(learned_config), "learned")
        baseline_contract = _scenario_contract_for(
            baseline_manifest, baseline_name, "external")
        if learned_contract["sha256"] != baseline_contract["sha256"]:
            raise ValueError(
                "learned and external scenario contract fingerprints differ")
        if scenario_contract_fingerprint(
                str(learned_config)) != learned_contract:
            raise ValueError(
                "current scenario contract does not match V6 evidence")
        for label, frame, contract in [
            ("learned", learned_all, learned_contract),
            ("external", baseline_all, baseline_contract),
        ]:
            observed_contracts = set(
                frame["scenario_contract_sha256"].astype(str))
            if observed_contracts != {str(contract["sha256"])}:
                raise ValueError(
                    f"{label} V6 rows disagree with scenario contract manifest")
        provenance["scenario_contract_sha256"] = learned_contract["sha256"]
        learned_config_records = learned_manifest.get(
            "config_fingerprints", {})
        baseline_config_records = baseline_manifest.get(
            "config_fingerprints", {})
        if str(learned_config) not in learned_config_records:
            raise ValueError("learned V6 config fingerprint is missing")
        if baseline_name not in baseline_config_records:
            raise ValueError("external V6 config fingerprint is missing")
        if (str(learned_config) == baseline_name
                and learned_config_records[str(learned_config)]
                != baseline_config_records[baseline_name]):
            raise ValueError(
                "learned and external config fingerprints differ")
    else:
        selected_methods = (
            methods or sorted(baselines["method"].astype(str).unique()))
    pair_frames = []
    summary_rows = []
    for method in selected_methods:
        baseline = baselines[baselines["method"].astype(str).eq(method)].copy()
        if baseline.empty:
            raise ValueError(f"missing external baseline method {method}")
        if baseline.duplicated(["eval_seed"]).any():
            raise ValueError(f"{method}: duplicate rows per eval seed")
        _validate_tapes(learned, baseline, method)
        merged = learned.merge(
            baseline,
            on="eval_seed",
            how="inner",
            suffixes=("_learned", "_baseline"),
            validate="many_to_one",
        )
        expected = len(learned)
        if len(merged) != expected:
            raise ValueError(
                f"{method}: paired rows {len(merged)}, expected {expected}")
        pair_frame = merged[["train_seed", "eval_seed"]].copy()
        pair_frame["learned_config"] = learned_config
        pair_frame["baseline_config"] = baseline_name
        pair_frame["baseline_method"] = method
        row = {
            "learned_config": learned_config,
            "baseline_config": baseline_name,
            "baseline_method": method,
            "n_train_seeds": int(learned["train_seed"].nunique()),
            "n_eval_seeds": int(learned["eval_seed"].nunique()),
            "n_pairs": int(len(merged)),
        }
        for metric in metrics:
            learned_values = pd.to_numeric(
                merged[f"{metric}_learned"], errors="raise")
            baseline_values = pd.to_numeric(
                merged[f"{metric}_baseline"], errors="raise")
            delta_name = f"delta_{metric}"
            pair_frame[delta_name] = learned_values - baseline_values
            bootstrap = hierarchical_bootstrap(pair_frame, delta_name)
            lo, hi = np.percentile(bootstrap, [2.5, 97.5])
            train_delta = pair_frame.groupby(
                "train_seed")[delta_name].mean().to_numpy(dtype=float)
            train_std = (
                float(train_delta.std(ddof=1)) if train_delta.size > 1 else 0.0
            )
            direction = METRIC_DIRECTIONS.get(metric, "descriptive")
            if direction == "min":
                probability_better = float(np.mean(bootstrap < 0.0))
                orientation = -1.0
            elif direction == "max":
                probability_better = float(np.mean(bootstrap > 0.0))
                orientation = 1.0
            else:
                probability_better = float("nan")
                orientation = 1.0
            row[f"{metric}_learned_mean"] = float(learned_values.mean())
            row[f"{metric}_baseline_mean"] = float(baseline_values.mean())
            row[f"{delta_name}_mean"] = float(pair_frame[delta_name].mean())
            row[f"{delta_name}_ci_low"] = float(lo)
            row[f"{delta_name}_ci_high"] = float(hi)
            row[f"{delta_name}_prob_learned_better"] = probability_better
            row[f"{delta_name}_paired_effect_dz"] = (
                orientation * float(train_delta.mean()) / train_std
                if train_std > 0.0 else float("nan")
            )
            row[f"{delta_name}_signflip_p"] = paired_sign_flip_p(train_delta)
        pair_frames.append(pair_frame)
        summary_rows.append(row)

    for metric in metrics:
        key = f"delta_{metric}_signflip_p"
        adjusted = holm_adjusted_pvalues([
            float(row.get(key, float("nan"))) for row in summary_rows
        ])
        for row, value in zip(summary_rows, adjusted):
            row[f"{key}_holm"] = value

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_output = pd.concat(pair_frames, ignore_index=True)
    summary_output = pd.DataFrame(summary_rows)
    pair_path = out_dir / "learned_vs_external_per_pair.csv"
    summary_path = out_dir / "learned_vs_external_summary.csv"
    pair_output.to_csv(pair_path, index=False)
    summary_output.to_csv(summary_path, index=False)
    manifest_payload = {
        "manifest_version": (
            V6_COMPARISON_MANIFEST_VERSION
            if protocol_version == V6_PROTOCOL_VERSION else None),
        "protocol_version": protocol_version,
        "learned_input": str(learned_path),
        "baseline_input": str(baseline_path),
        "learned_config": learned_config,
        "baseline_config": baseline_name,
        "baseline_methods": selected_methods,
        "metrics": metrics,
        "delta_definition": "learned minus external baseline",
        "uncertainty": (
            "paired crossed bootstrap over train and eval seed; each draw "
            "shares one evaluation-seed resample across training seeds"
        ),
        "paired_test": "two-sided sign-flip over train-seed mean deltas",
        "multiple_testing": "Holm correction across external methods per metric",
        "common_random_numbers_verified": True,
        "source_provenance": provenance,
    }
    if protocol_version == V6_PROTOCOL_VERSION:
        manifest_payload.update({
            "strict_complete": True,
            "required_external_method_family": list(
                REQUIRED_EXTERNAL_METHODS),
            "input_artifacts": {
                "learned": learned_manifest["artifacts"][
                    "frozen_per_eval.csv"],
                "external": baseline_manifest["artifacts"][
                    "external_baselines_per_seed.csv"],
            },
            "artifacts": {
                "learned_vs_external_per_pair.csv": csv_artifact_record(
                    pair_path, pair_output, V6_PAIR_PRIMARY_KEY),
                "learned_vs_external_summary.csv": csv_artifact_record(
                    summary_path,
                    summary_output,
                    V6_COMPARISON_SUMMARY_PRIMARY_KEY,
                ),
            },
        })
    (out_dir / "learned_vs_external_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--learned-config", required=True)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--learned-manifest", default=None)
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    methods = (
        [value.strip() for value in args.methods.split(",") if value.strip()]
        if args.methods else None
    )
    compare(
        learned_path=Path(args.learned),
        baseline_path=Path(args.baselines),
        out_dir=Path(args.out_dir),
        learned_config=args.learned_config,
        baseline_config=args.baseline_config,
        methods=methods,
        learned_manifest_path=(
            Path(args.learned_manifest) if args.learned_manifest else None),
        baseline_manifest_path=(
            Path(args.baseline_manifest) if args.baseline_manifest else None),
    )


if __name__ == "__main__":
    main()
