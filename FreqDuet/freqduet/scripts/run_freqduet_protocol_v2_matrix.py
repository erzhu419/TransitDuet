#!/usr/bin/env python3
"""Train protocol-v2 policies and evaluate frozen checkpoints on new scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from scripts.analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        files_fingerprint,
        runtime_environment,
        sha256_file,
        validate_csv_artifact,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        files_fingerprint,
        runtime_environment,
        sha256_file,
        validate_csv_artifact,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "F_freqduet_protocol_v2_main_hiro",
    "F_freqduet_protocol_v2_upperdisc_hiro",
    "F_freqduet_protocol_v2_upperhist_hiro",
    "F_freqduet_protocol_v2_upperdisc_hist_hiro",
]
DEFAULT_TRAIN_SEEDS = [7, 17, 31, 42]
DEFAULT_EVAL_SEEDS = [10001, 10007, 10009, 10037, 10039, 10061, 10067, 10069]
METRICS = [
    "service_cost",
    "avg_wait_observed_min",
    "restricted_wait_horizon_min",
    "passenger_unserved_rate",
    "headway_cv",
    "fleet_overshoot",
    "trip_launch_rate",
    "trip_completion_rate",
    "terminal_launch_shift_mean",
    "upper_delta_mean",
    "lower_action_mean",
]
V4_SAFETY_METRICS = [
    "avg_in_vehicle_observed_min",
    "restricted_in_vehicle_horizon_min",
    "avg_total_journey_observed_min",
    "restricted_total_journey_horizon_min",
    "holding_vehicle_seconds",
    "holding_passenger_seconds",
    "fleet_denied_dispatch_events",
    "fleet_denied_trips",
    "fleet_readiness_delay_mean_s",
    "fleet_readiness_delay_max_s",
]
V5_SAFETY_METRICS = V4_SAFETY_METRICS + [
    "holding_vehicle_seconds_per_launched_trip",
    "holding_passenger_min_per_generated",
    "fleet_denied_trip_rate",
]
V5_MECHANISM_METRICS = [
    "lower_causal_guard_enabled",
    "lower_causal_guard_active_mean",
    "lower_causal_guard_limit_mean_s",
    "lower_causal_guard_adjustment_mean_s",
    "upper_interval_wait_cost_sum",
    "upper_interval_onboard_cost_sum",
    "upper_interval_dispatch_backlog_cost_sum",
    "upper_plan_raw_delta_mean_s",
    "upper_plan_projected_delta_mean_s",
    "upper_plan_projected_delta_sum_abs_mean_s",
]
V6_SAFETY_METRICS = V5_SAFETY_METRICS + [
    "fleet_denied_retry_trip_seconds",
    "commanded_holding_vehicle_seconds",
    "commanded_holding_passenger_seconds",
    "commanded_holding_passenger_min_per_generated",
    "terminal_actual_dispatch_gap_mean_s",
    "terminal_dispatch_execution_error_mean_s",
    "terminal_dispatch_execution_error_abs_mean_s",
]
V6_MECHANISM_METRICS = list(V5_MECHANISM_METRICS)
V6_CONTRACT_COLUMNS = ["lower_causal_guard_evidence_mode"]
OPTIONAL_COST_METRICS = [
    "service_cost_observed",
    "service_cost_restricted",
]
METRIC_DIRECTIONS = {
    "service_cost": "min",
    "service_cost_observed": "min",
    "service_cost_restricted": "min",
    "avg_wait_observed_min": "min",
    "restricted_wait_horizon_min": "min",
    "passenger_unserved_rate": "min",
    "headway_cv": "min",
    "fleet_overshoot": "min",
    "trip_launch_rate": "max",
    "trip_completion_rate": "max",
    "avg_in_vehicle_observed_min": "min",
    "restricted_in_vehicle_horizon_min": "min",
    "avg_total_journey_observed_min": "min",
    "restricted_total_journey_horizon_min": "min",
    "holding_vehicle_seconds": "min",
    "holding_passenger_seconds": "min",
    "fleet_denied_dispatch_events": "min",
    "fleet_denied_trips": "min",
    "fleet_readiness_delay_mean_s": "min",
    "fleet_readiness_delay_max_s": "min",
    "holding_vehicle_seconds_per_launched_trip": "min",
    "holding_passenger_min_per_generated": "min",
    "fleet_denied_trip_rate": "min",
    "upper_interval_wait_cost_sum": "min",
    "upper_interval_onboard_cost_sum": "min",
    "upper_interval_dispatch_backlog_cost_sum": "min",
    "fleet_denied_retry_trip_seconds": "min",
    "commanded_holding_vehicle_seconds": "min",
    "commanded_holding_passenger_seconds": "min",
    "commanded_holding_passenger_min_per_generated": "min",
    "terminal_actual_dispatch_gap_mean_s": "descriptive",
    "terminal_dispatch_execution_error_mean_s": "descriptive",
    "terminal_dispatch_execution_error_abs_mean_s": "min",
}


def strict_protocol_metrics(protocol_version: str) -> list[str]:
    if str(protocol_version) == "freqduet-eval-v4":
        return list(V4_SAFETY_METRICS)
    if str(protocol_version) == "freqduet-eval-v5":
        return list(V5_SAFETY_METRICS + V5_MECHANISM_METRICS)
    if str(protocol_version) == "freqduet-eval-v6":
        return list(V6_SAFETY_METRICS + V6_MECHANISM_METRICS)
    return []


def analysis_metrics_for_frame(frame: pd.DataFrame) -> list[str]:
    """Select explicit cost views only when the whole matrix provides both."""
    optional_any = {
        metric: metric in frame.columns and frame[metric].notna().any()
        for metric in OPTIONAL_COST_METRICS
    }
    optional_complete = {
        metric: metric in frame.columns and frame[metric].notna().all()
        for metric in OPTIONAL_COST_METRICS
    }
    if (len(set(optional_any.values())) > 1
            or optional_any != optional_complete):
        raise RuntimeError(
            "matrix contains incomplete explicit service-cost views: "
            f"any={optional_any}, complete={optional_complete}")
    wait_bases = set()
    if "service_cost_wait_metric" in frame.columns:
        wait_bases = {
            str(value).strip().lower()
            for value in frame["service_cost_wait_metric"].dropna().tolist()
            if str(value).strip()
        }
    mixed_wait_basis = len(wait_bases) > 1
    if mixed_wait_basis and not all(optional_complete.values()):
        raise RuntimeError(
            "matrix mixes generic service-cost wait bases without complete "
            f"explicit cost views: {sorted(wait_bases)}")
    metrics = [
        metric for metric in METRICS
        if metric != "service_cost" or not mixed_wait_basis
    ]
    protocol_versions = set(frame.get(
        "protocol_version", pd.Series(dtype=str)).astype(str))
    for protocol_version in sorted(protocol_versions):
        strict_metrics = strict_protocol_metrics(protocol_version)
        if not strict_metrics:
            continue
        missing = sorted(set(strict_metrics) - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"{protocol_version} matrix is missing required passenger/"
                f"fleet/mechanism safety metrics: {missing}")
        incomplete = [
            metric for metric in strict_metrics
            if not pd.to_numeric(frame[metric], errors="coerce").notna().all()
        ]
        if incomplete:
            raise RuntimeError(
                f"{protocol_version} matrix has incomplete required metrics: "
                f"{incomplete}")
        metrics.extend(metric for metric in strict_metrics if metric not in metrics)
    if all(optional_complete.values()):
        metrics.extend(OPTIONAL_COST_METRICS)
    return metrics
PROTOCOL_VERSION = "freqduet-eval-v2"
RUN_MANIFEST_VERSION = "freqduet-run-manifest-v2"
LEGACY_RUN_MANIFEST_VERSION = "freqduet-run-manifest-v1"
RUN_MANIFEST_NAME = "protocol_run_manifest.json"
MATRIX_MANIFEST_VERSION = "freqduet-matrix-manifest-v2"
MODEL_SOURCE_FINGERPRINT_VERSION = "freqduet-model-source-v2"
SOURCE_PACKAGE_DIRS = ["env", "frequency", "lower", "upper", "coupling"]
SOURCE_FIXED_FILES = [
    "runner_v3.py",
    "randomness.py",
    "env/config.json",
    "env/data/passenger_OD.xlsx",
    "env/data/route_news.xlsx",
    "env/data/stop_news.xlsx",
    "env/data/time_table.xlsx",
]
ANALYSIS_FILES = [
    "scripts/analysis_provenance.py",
    "scripts/run_freqduet_protocol_v2_matrix.py",
    "scripts/decide_freqduet_protocol_v6_screen.py",
    "scripts/compare_freqduet_external_frozen.py",
    "scripts/run_freqduet_external_baselines.py",
    "scripts/submit_freqduet_protocol_v2_scheduleurm.py",
    "scripts/validate_freqduet_protocol_v6_configs.py",
]
SCENARIO_SOURCE_FILES = [
    "randomness.py",
    *sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "env").glob("*.py")
    ),
]
MATRIX_STAGES = {"development", "confirmation", "exploratory"}


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def config_path(name: str) -> Path:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return ROOT / "configs_freqduet" / filename


def config_name(name: str) -> str:
    return Path(name).stem


def run_dir(logs_dir: Path, config: str, train_seed: int) -> Path:
    return logs_dir / f"{config_name(config)}_seed{int(train_seed)}"


def worker_env(worker_threads: int, suppress_heavy_artifacts: bool) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(worker_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "FREQDUET_TORCH_THREADS",
    ]:
        env[key] = threads
    if suppress_heavy_artifacts:
        env["FREQDUET_SUPPRESS_HEAVY_ARTIFACTS"] = "1"
    return env


def training_complete(
    path: Path,
    episodes: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> bool:
    diagnostics = path / "diagnostics.csv"
    checkpoints = path / "checkpoints"
    if not diagnostics.exists() or not checkpoints.exists():
        return False
    try:
        frame = pd.read_csv(diagnostics)
    except Exception:
        return False
    expected_episodes = set(range(int(episodes)))
    actual_episodes = set(frame["ep"].astype(int).tolist())
    final_ep = int(episodes) - 1
    return (
        len(frame) == int(episodes)
        and actual_episodes == expected_episodes
        and "protocol_version" in frame
        and set(frame["protocol_version"].astype(str)) == {
            str(protocol_version)}
        and (checkpoints / f"lower_ep{final_ep}.pt").exists()
        and (checkpoints / f"upper_ep{final_ep}.pt").exists()
        and (checkpoints / f"runner_ep{final_ep}.pt").exists()
    )


def validate_evaluation_frame(
    frame: pd.DataFrame,
    eval_seeds: list[int],
    source: Path | str,
    checkpoint_ep: int | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    expected = [int(seed) for seed in eval_seeds]
    if len(expected) != len(set(expected)):
        raise ValueError("evaluation seeds must be unique")
    required_metrics = list(METRICS)
    required_metrics.extend(strict_protocol_metrics(protocol_version))
    required = {
        "protocol_version", "eval_seed", "checkpoint_ep", "policy_digest",
        "scenario_tape_id", *required_metrics,
    }
    if str(protocol_version) == "freqduet-eval-v6":
        required.update(V6_CONTRACT_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing evaluation columns {missing}")
    actual = frame["eval_seed"].astype(int)
    if len(frame) != len(expected) or actual.duplicated().any():
        raise ValueError(
            f"{source}: expected one row per evaluation seed")
    if set(actual) != set(expected):
        raise ValueError(
            f"{source}: evaluation seed set does not match manifest")
    if set(frame["protocol_version"].astype(str)) != {str(protocol_version)}:
        raise ValueError(f"{source}: protocol version mismatch")
    if frame["policy_digest"].astype(str).nunique() != 1:
        raise ValueError(f"{source}: policy digest changed within evaluation")
    if frame["scenario_tape_id"].isna().any():
        raise ValueError(f"{source}: missing scenario tape identifier")
    checkpoint_values = set(frame["checkpoint_ep"].astype(int))
    if len(checkpoint_values) != 1:
        raise ValueError(f"{source}: multiple checkpoint episodes")
    if checkpoint_ep is not None and checkpoint_values != {int(checkpoint_ep)}:
        raise ValueError(f"{source}: checkpoint episode mismatch")
    for metric in required_metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: non-finite metric {metric}")
    if str(protocol_version) == "freqduet-eval-v6":
        evidence = set(
            frame["lower_causal_guard_evidence_mode"].astype(str))
        if evidence != {"pre_action_departure_v6"}:
            raise ValueError(
                f"{source}: V6 holding guard evidence contract mismatch: "
                f"{sorted(evidence)}")


def evaluation_complete(
    path: Path,
    eval_seeds: list[int],
    checkpoint_ep: int | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> bool:
    result = path / "frozen_evaluation" / "evaluation.csv"
    manifest_path = result.parent / "evaluation_manifest.json"
    if not result.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("manifest_version") != (
                "freqduet-evaluation-manifest-v2"):
            return False
        frame = validate_csv_artifact(
            result,
            (manifest.get("artifacts") or {}).get("evaluation_csv"),
            expected_primary_key=["eval_seed"],
        )
        validate_evaluation_frame(
            frame,
            eval_seeds,
            result,
            checkpoint_ep=checkpoint_ep,
            protocol_version=protocol_version,
        )
        if manifest.get("protocol_version") != str(protocol_version):
            return False
        if [int(seed) for seed in manifest.get("scenario_seeds", [])] \
                != [int(seed) for seed in eval_seeds]:
            return False
        if str(manifest.get("policy_digest")) != str(
                frame["policy_digest"].iloc[0]):
            return False
        if (checkpoint_ep is not None
                and int(manifest.get("checkpoint_ep", -1))
                != int(checkpoint_ep)):
            return False
    except Exception:
        return False
    return True


def validate_evaluation_manifest(
    result: Path,
    frame: pd.DataFrame,
    config: str,
    train_seed: int,
    eval_seeds: list[int],
    checkpoint_ep: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    manifest_path = result.parent / "evaluation_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{result}: missing evaluation manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_version") != "freqduet-evaluation-manifest-v2":
        raise ValueError(f"{manifest_path}: manifest version mismatch")
    expected = {
        "protocol_version": str(protocol_version),
        "config_name": config_name(config),
        "training_seed": int(train_seed),
        "scenario_seeds": [int(seed) for seed in eval_seeds],
        "n_episodes": len(eval_seeds),
        "checkpoint_ep": int(checkpoint_ep),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{manifest_path}: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    if str(manifest.get("policy_digest")) != str(
            frame["policy_digest"].iloc[0]):
        raise ValueError(f"{manifest_path}: policy digest mismatch")
    locked_frame = validate_csv_artifact(
        result,
        (manifest.get("artifacts") or {}).get("evaluation_csv"),
        expected_primary_key=["eval_seed"],
    )
    if not locked_frame.equals(frame):
        raise ValueError(f"{manifest_path}: evaluation CSV changed while read")


def run_subprocess(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout or "")
    if process.returncode != 0:
        tail = "\n".join((process.stdout or "").splitlines()[-60:])
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{tail}")


def run_job(
    config: str,
    train_seed: int,
    eval_seeds: list[int],
    train_episodes: int,
    logs_dir: Path,
    worker_threads: int,
    suppress_heavy_artifacts: bool,
    clean: bool,
    skip_existing: bool,
    stage: str = "development",
) -> tuple[str, int, Path]:
    path = run_dir(logs_dir, config, train_seed)
    protocol_version = protocol_version_for_config(config)
    if clean and path.exists():
        shutil.rmtree(path)
    expected_manifest = run_manifest(
        config=config,
        train_seed=train_seed,
        eval_seeds=eval_seeds,
        train_episodes=train_episodes,
        stage=stage,
    )
    manifest_path = path / RUN_MANIFEST_NAME
    existing_artifacts = (
        (path / "diagnostics.csv").exists()
        or (path / "frozen_evaluation/evaluation.csv").exists()
    )
    if skip_existing and existing_artifacts:
        validate_run_manifest(
            manifest_path,
            expected=expected_manifest,
            source=path,
        )
    path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n")
    env = worker_env(worker_threads, suppress_heavy_artifacts)
    if not (skip_existing and training_complete(
            path, train_episodes, protocol_version=protocol_version)):
        train_command = [
            sys.executable,
            "runner_v3.py",
            "--config",
            str(config_path(config)),
            "--episodes",
            str(int(train_episodes)),
            "--seed",
            str(int(train_seed)),
            "--no-resume",
            "--logs-dir",
            str(logs_dir),
        ]
        run_subprocess(train_command, ROOT, env, path / "train_stdout.log")

    if not training_complete(
            path, train_episodes, protocol_version=protocol_version):
        raise RuntimeError(f"incomplete training artifacts in {path}")
    final_ep = int(train_episodes) - 1
    if not (skip_existing and evaluation_complete(
            path,
            eval_seeds,
            checkpoint_ep=final_ep,
            protocol_version=protocol_version)):
        eval_command = [
            sys.executable,
            "runner_v3.py",
            "--config",
            str(config_path(config)),
            "--seed",
            str(int(train_seed)),
            "--eval-only",
            "--checkpoint-dir",
            str(path / "checkpoints"),
            "--checkpoint-ep",
            str(final_ep),
            "--eval-seeds",
            ",".join(str(seed) for seed in eval_seeds),
            "--eval-output-dir",
            str(path / "frozen_evaluation"),
            "--logs-dir",
            str(logs_dir),
        ]
        run_subprocess(eval_command, ROOT, env, path / "eval_stdout.log")
    if not evaluation_complete(
            path,
            eval_seeds,
            checkpoint_ep=final_ep,
            protocol_version=protocol_version):
        raise RuntimeError(f"incomplete frozen evaluation in {path}")
    return config_name(config), int(train_seed), path


def selected_jobs(
    configs: list[str],
    train_seeds: list[int],
    start: int | None,
    end: int | None,
) -> list[tuple[str, int]]:
    jobs = [(config, seed) for config in configs for seed in train_seeds]
    lo = 0 if start is None else max(0, int(start))
    hi = len(jobs) if end is None else min(len(jobs), int(end))
    return jobs[lo:max(lo, hi)]


def hierarchical_bootstrap(
    frame: pd.DataFrame, metric: str, draws: int = 5000
) -> np.ndarray:
    """Bootstrap a complete train-seed x evaluation-seed crossed design.

    Evaluation scenarios are shared across every training seed, so the same
    resampled evaluation indices must be used for all resampled policies in a
    draw. Treating evaluation seeds as nested independently under each policy
    destroys that crossed dependence and understates common scenario effects.
    """
    required = {"train_seed", "eval_seed", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"crossed bootstrap missing columns: {missing}")
    if frame.duplicated(["train_seed", "eval_seed"]).any():
        raise ValueError("crossed bootstrap requires one row per train/eval pair")
    table = frame.pivot(
        index="train_seed", columns="eval_seed", values=metric).sort_index(
            axis=0).sort_index(axis=1)
    if table.empty or table.isna().any().any():
        raise ValueError(
            "crossed bootstrap requires a complete train-seed x eval-seed grid")
    values = table.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"crossed bootstrap metric {metric} is non-finite")
    rng = np.random.RandomState(20260803)
    n_train, n_eval = values.shape
    sampled_train = rng.randint(0, n_train, size=(int(draws), n_train))
    sampled_eval = rng.randint(0, n_eval, size=(int(draws), n_eval))
    sampled = values[
        sampled_train[:, :, None], sampled_eval[:, None, :]]
    return sampled.mean(axis=(1, 2))


def validate_common_scenario_tapes(per_eval: pd.DataFrame) -> None:
    """Require one immutable scenario tape per evaluation seed globally."""
    required = {"eval_seed", "scenario_tape_id"}
    missing = sorted(required - set(per_eval.columns))
    if missing:
        raise ValueError(f"scenario-tape validation missing columns: {missing}")
    if per_eval["scenario_tape_id"].isna().any():
        raise ValueError("matrix contains missing scenario tape identifiers")
    tape_counts = per_eval.groupby("eval_seed")["scenario_tape_id"].nunique()
    if not tape_counts.eq(1).all():
        bad = tape_counts[~tape_counts.eq(1)].index.astype(int).tolist()
        raise RuntimeError(
            "policies did not use one common scenario tape for eval seeds: "
            f"{bad}")


def hierarchical_interval(
    frame: pd.DataFrame, metric: str, draws: int = 5000
) -> tuple[float, float]:
    estimates = hierarchical_bootstrap(frame, metric, draws=draws)
    return tuple(
        float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def paired_sign_flip_p(values: np.ndarray, max_draws: int = 65536) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    total_exact = 1 << int(values.size)
    exact = total_exact <= int(max_draws)
    if exact:
        indices = np.arange(total_exact, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(values.size, dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
    else:
        rng = np.random.RandomState(20260803)
        signs = rng.choice(
            [-1.0, 1.0], size=(int(max_draws), values.size))
    null_means = np.abs((signs * values[None, :]).mean(axis=1))
    count = int(np.count_nonzero(null_means >= observed))
    if exact:
        return float(count / len(null_means))
    return float((count + 1) / (len(null_means) + 1))


def holm_adjusted_pvalues(values: list[float]) -> list[float]:
    """Control family-wise error while preserving the original row order."""
    raw = np.asarray(values, dtype=np.float64)
    adjusted = np.full(raw.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(raw))
    if finite_indices.size == 0:
        return adjusted.tolist()
    order = finite_indices[np.argsort(raw[finite_indices], kind="stable")]
    family_size = len(order)
    running_max = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, float(raw[index]) * (family_size - rank))
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted.tolist()


def _resolve_parent(path: Path, parent: str) -> Path:
    parent_path = Path(parent)
    if parent_path.is_absolute():
        return parent_path.resolve()
    candidates = [
        path.parent / parent_path,
        path.parent.parent / parent_path,
        ROOT / parent_path,
    ]
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[-1].resolve(),
    )


def config_lineage(path: Path, seen: set[Path] | None = None) -> list[Path]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    lineage = []
    if "_extends" in payload:
        lineage.extend(config_lineage(
            _resolve_parent(path, str(payload["_extends"])), seen))
    lineage.append(path)
    return lineage


def config_fingerprint(name: str) -> dict[str, object]:
    lineage = config_lineage(config_path(name))
    digest = hashlib.sha256()
    labels = []
    for path in lineage:
        try:
            label = str(path.relative_to(ROOT))
        except ValueError:
            label = str(path)
        labels.append(label)
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "lineage": labels}


def resolved_config(name: str) -> dict[str, object]:
    """Resolve one YAML inheritance chain without importing the training runner."""
    result: dict[str, object] = {}

    def merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if (key in base and isinstance(base[key], dict)
                    and isinstance(value, dict)):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    for path in config_lineage(config_path(name)):
        payload = yaml.safe_load(path.read_text()) or {}
        payload.pop("_extends", None)
        merge(result, payload)
    return result


def protocol_version_for_config(name: str) -> str:
    payload = resolved_config(name)
    protocol = payload.get("protocol", {}) or {}
    return str(protocol.get("version", PROTOCOL_VERSION))


def analysis_fingerprint() -> dict[str, object]:
    return files_fingerprint(ROOT, ANALYSIS_FILES)


def scenario_contract(name: str) -> dict[str, object]:
    """Fingerprint exogenous dynamics shared by every policy in a matrix."""
    payload = resolved_config(name)
    protocol = payload.get("protocol", {}) or {}
    contract_payload = {
        "contract_version": "freqduet-scenario-contract-v1",
        "protocol_version": str(protocol.get("version", PROTOCOL_VERSION)),
        "objective_contract": str(protocol.get("objective_contract", "")),
        "env": payload.get("env", {}) or {},
        "randomness": payload.get("randomness", {}) or {},
        "scenario_sources": files_fingerprint(ROOT, SCENARIO_SOURCE_FILES),
        "environment_data": files_fingerprint(ROOT, [
            "env/config.json",
            "env/data/passenger_OD.xlsx",
            "env/data/route_news.xlsx",
            "env/data/stop_news.xlsx",
            "env/data/time_table.xlsx",
        ]),
    }
    return {
        "version": "freqduet-scenario-contract-v1",
        "sha256": canonical_json_sha256(contract_payload),
        "payload": contract_payload,
    }


def source_fingerprint() -> dict[str, object]:
    paths = {ROOT / relative for relative in SOURCE_FIXED_FILES}
    for package in SOURCE_PACKAGE_DIRS:
        paths.update((ROOT / package).rglob("*.py"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"source fingerprint inputs missing: {missing}")

    digest = hashlib.sha256()
    files = []
    for path in sorted(paths):
        label = str(path.relative_to(ROOT))
        payload = path.read_bytes()
        file_sha = hashlib.sha256(payload).hexdigest()
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha))
        files.append({
            "path": label,
            "sha256": file_sha,
            "size_bytes": len(payload),
        })
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def run_manifest(
    *, config: str, train_seed: int, eval_seeds: list[int],
    train_episodes: int, stage: str = "development",
) -> dict[str, object]:
    if str(stage) not in MATRIX_STAGES:
        raise ValueError(f"unsupported matrix stage {stage!r}")
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "model_source_fingerprint_version": (
            MODEL_SOURCE_FINGERPRINT_VERSION),
        "protocol_version": protocol_version_for_config(config),
        "stage": str(stage),
        "config_name": config_name(config),
        "config_fingerprint": config_fingerprint(config),
        "source_fingerprint": source_fingerprint(),
        "scenario_contract": scenario_contract(config),
        "analysis_fingerprint_at_launch": analysis_fingerprint(),
        "runtime_environment": runtime_environment(),
        "invocation": {
            "entrypoint": "scripts/run_freqduet_protocol_v2_matrix.py",
            "argv": [str(value) for value in sys.argv],
            "cwd": str(ROOT),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "git": git_provenance(),
        "train_seed": int(train_seed),
        "train_episodes": int(train_episodes),
        "checkpoint_ep": int(train_episodes) - 1,
        "eval_seeds": [int(seed) for seed in eval_seeds],
    }


def validate_run_manifest(
    path: Path,
    *,
    expected: dict[str, object] | None = None,
    source: Path | str | None = None,
) -> dict[str, object]:
    label = source or path
    if not path.exists():
        raise ValueError(f"{label}: missing {RUN_MANIFEST_NAME}")
    payload = json.loads(path.read_text())
    if payload.get("manifest_version") != RUN_MANIFEST_VERSION:
        raise ValueError(f"{label}: run manifest version mismatch")
    if expected is not None and payload != expected:
        raise ValueError(
            f"{label}: existing artifacts do not match the current source, "
            "config, seeds, or episode protocol")
    source_payload = payload.get("source_fingerprint", {})
    if len(str(source_payload.get("sha256", ""))) != 64:
        raise ValueError(f"{label}: invalid source fingerprint")
    if payload.get("model_source_fingerprint_version") != (
            MODEL_SOURCE_FINGERPRINT_VERSION):
        raise ValueError(f"{label}: model source fingerprint version mismatch")
    scenario_payload = payload.get("scenario_contract", {})
    if len(str(scenario_payload.get("sha256", ""))) != 64:
        raise ValueError(f"{label}: invalid scenario contract fingerprint")
    if not isinstance(payload.get("runtime_environment"), dict):
        raise ValueError(f"{label}: missing runtime environment")
    if not isinstance(payload.get("invocation"), dict):
        raise ValueError(f"{label}: missing invocation provenance")
    return payload


def git_provenance() -> dict[str, object]:
    def run(*args: str) -> tuple[bool, str]:
        try:
            process = subprocess.run(
                ["git", *args], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            return False, ""
        return process.returncode == 0, process.stdout.strip()

    commit_ok, commit = run("rev-parse", "HEAD")
    branch_ok, branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status_ok, status = run(
        "status", "--porcelain", "--untracked-files=no")
    if not commit_ok:
        commit = str(os.environ.get(
            "FREQDUET_SOURCE_COMMIT", "unavailable")).strip()
    if not branch_ok:
        branch = str(os.environ.get(
            "FREQDUET_SOURCE_BRANCH", "unavailable")).strip()
    if status_ok:
        tracked_dirty: bool | None = bool(status)
    else:
        dirty_env = os.environ.get("FREQDUET_SOURCE_TRACKED_DIRTY")
        tracked_dirty = (
            None if dirty_env is None
            else str(dirty_env).strip().lower() in {"1", "true", "yes"}
        )

    return {
        "commit": commit or "unavailable",
        "branch": branch or "unavailable",
        "tracked_dirty": tracked_dirty,
    }


def aggregate(
    configs: list[str], train_seeds: list[int], eval_seeds: list[int],
    logs_dirs: list[Path], out_dir: Path, reference: str,
    train_episodes: int, stage: str,
) -> None:
    if str(stage) not in MATRIX_STAGES:
        raise ValueError(f"unsupported matrix stage {stage!r}")
    if int(train_episodes) <= 0:
        raise ValueError("train_episodes must be positive")
    checkpoint_ep = int(train_episodes) - 1
    protocol_versions = {
        protocol_version_for_config(config) for config in configs}
    if len(protocol_versions) != 1:
        raise ValueError(
            "one matrix cannot mix protocol versions: "
            f"{sorted(protocol_versions)}")
    protocol_version = next(iter(protocol_versions))
    expected_source = source_fingerprint()
    scenario_contracts = {
        config_name(config): scenario_contract(config) for config in configs
    }
    scenario_hashes = {
        str(value["sha256"]) for value in scenario_contracts.values()
    }
    if len(scenario_hashes) != 1:
        raise ValueError(
            "one matrix cannot mix scenario contracts: "
            f"{scenario_contracts}")
    common_scenario_contract = next(iter(scenario_contracts.values()))
    records = []
    missing_runs = []
    run_manifests = []
    run_manifest_artifacts = []
    for config in configs:
        name = config_name(config)
        for train_seed in train_seeds:
            candidates = [
                run_dir(logs_dir, config, train_seed)
                / "frozen_evaluation/evaluation.csv"
                for logs_dir in logs_dirs
            ]
            candidates = [path for path in candidates if path.exists()]
            if not candidates:
                missing_runs.append((name, int(train_seed)))
                continue
            if len(candidates) > 1:
                raise RuntimeError(
                    f"duplicate evaluation artifacts for {name} seed "
                    f"{train_seed}: {candidates}")
            result = candidates[0]
            frame = pd.read_csv(result)
            validate_evaluation_frame(
                frame,
                eval_seeds,
                result,
                checkpoint_ep=checkpoint_ep,
                protocol_version=protocol_version,
            )
            validate_evaluation_manifest(
                result,
                frame,
                config,
                train_seed,
                eval_seeds,
                checkpoint_ep=checkpoint_ep,
                protocol_version=protocol_version,
            )
            run_manifest_path = result.parent.parent / RUN_MANIFEST_NAME
            manifest = validate_run_manifest(
                run_manifest_path, source=result.parent.parent)
            expected_fields = {
                "protocol_version": protocol_version,
                "stage": str(stage),
                "config_name": name,
                "config_fingerprint": config_fingerprint(config),
                "source_fingerprint": expected_source,
                "scenario_contract": scenario_contracts[name],
                "train_seed": int(train_seed),
                "train_episodes": int(train_episodes),
                "checkpoint_ep": checkpoint_ep,
                "eval_seeds": [int(seed) for seed in eval_seeds],
            }
            for key, expected_value in expected_fields.items():
                if manifest.get(key) != expected_value:
                    raise ValueError(
                        f"{run_manifest_path}: {key} does not match "
                        "the aggregation protocol")
            run_manifests.append(manifest)
            run_manifest_artifacts.append({
                "config": name,
                "train_seed": int(train_seed),
                "path": str(run_manifest_path),
                "sha256": sha256_file(run_manifest_path),
                "size_bytes": int(run_manifest_path.stat().st_size),
            })
            frame["config"] = name
            frame["train_seed"] = int(train_seed)
            frame["scenario_contract_sha256"] = str(
                common_scenario_contract["sha256"])
            records.append(frame)
    if missing_runs:
        raise RuntimeError(f"missing frozen evaluations: {missing_runs}")
    if not records:
        raise RuntimeError("no frozen evaluation files found")
    run_source_hashes = {
        manifest["source_fingerprint"]["sha256"]
        for manifest in run_manifests
    }
    if run_source_hashes != {str(expected_source["sha256"])}:
        raise RuntimeError(
            "matrix source fingerprint does not match the locked source: "
            f"{sorted(run_source_hashes)}")
    launch_analysis_hashes = {
        str(manifest.get("analysis_fingerprint_at_launch", {}).get(
            "sha256", ""))
        for manifest in run_manifests
    }
    if len(launch_analysis_hashes) != 1 or any(
            len(value) != 64 for value in launch_analysis_hashes):
        raise RuntimeError(
            "matrix mixes or lacks launch analysis fingerprints: "
            f"{sorted(launch_analysis_hashes)}")
    current_analysis = analysis_fingerprint()
    if (protocol_version == "freqduet-eval-v6"
            and launch_analysis_hashes != {
                str(current_analysis["sha256"])}):
        raise RuntimeError(
            "V6 analysis source changed between launch and aggregation")
    run_git_commits = {
        str((manifest.get("git") or {}).get("commit", ""))
        for manifest in run_manifests
    }
    run_git_dirty = {
        (manifest.get("git") or {}).get("tracked_dirty")
        for manifest in run_manifests
    }
    if (len(run_git_commits) != 1
            or any(not re.fullmatch(r"[0-9a-f]{40}", value)
                   for value in run_git_commits)
            or len(run_git_dirty) != 1
            or not isinstance(next(iter(run_git_dirty)), bool)):
        raise RuntimeError(
            "matrix mixes or lacks run Git provenance: "
            f"commits={sorted(run_git_commits)}, dirty={run_git_dirty}")
    aggregate_git = git_provenance()
    if (str(aggregate_git.get("commit", ""))
            != next(iter(run_git_commits))):
        raise RuntimeError(
            "aggregation Git commit differs from the run manifests")
    common_run_git = {
        "commit": next(iter(run_git_commits)),
        "tracked_dirty": next(iter(run_git_dirty)),
    }
    per_eval = pd.concat(records, ignore_index=True)
    expected_rows = len(configs) * len(train_seeds) * len(eval_seeds)
    if len(per_eval) != expected_rows:
        raise RuntimeError(
            f"matrix has {len(per_eval)} rows, expected {expected_rows}")
    if per_eval.duplicated(["config", "train_seed", "eval_seed"]).any():
        raise RuntimeError("matrix contains duplicate config/train/eval rows")
    observed_checkpoint = set(per_eval["checkpoint_ep"].astype(int))
    if observed_checkpoint != {checkpoint_ep}:
        raise RuntimeError(
            "matrix checkpoint does not match the locked training length")
    validate_common_scenario_tapes(per_eval)
    analysis_metrics = analysis_metrics_for_frame(per_eval)
    summary_rows = []
    for name, frame in per_eval.groupby("config", sort=False):
        row = {
            "config": name,
            "n_train_seeds": int(frame["train_seed"].nunique()),
            "n_eval_seeds": int(frame["eval_seed"].nunique()),
            "n_rollouts": int(len(frame)),
        }
        for metric in analysis_metrics:
            values = frame[metric].astype(float)
            lo, hi = hierarchical_interval(frame, metric)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        summary_rows.append(row)

    reference_name = config_name(reference)
    reference_frame = per_eval[per_eval["config"].eq(reference_name)]
    delta_rows = []
    for name in sorted(per_eval["config"].unique()):
        if name == reference_name:
            continue
        candidate = per_eval[per_eval["config"].eq(name)]
        merged = candidate.merge(
            reference_frame,
            on=["train_seed", "eval_seed"],
            suffixes=("_candidate", "_reference"),
        )
        if merged.empty:
            continue
        delta_frame = merged[["train_seed", "eval_seed"]].copy()
        row = {
            "candidate": name,
            "reference": reference_name,
            "n_pairs": int(len(merged)),
        }
        for metric in analysis_metrics:
            delta_col = f"delta_{metric}"
            delta_frame[delta_col] = (
                merged[f"{metric}_candidate"].astype(float)
                - merged[f"{metric}_reference"].astype(float)
            )
            bootstrap = hierarchical_bootstrap(delta_frame, delta_col)
            lo, hi = tuple(float(value) for value in np.percentile(
                bootstrap, [2.5, 97.5]))
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
            train_deltas = delta_frame.groupby(
                "train_seed")[delta_col].mean().to_numpy(dtype=float)
            train_std = float(train_deltas.std(ddof=1)) \
                if train_deltas.size > 1 else 0.0
            row[f"{delta_col}_mean"] = float(delta_frame[delta_col].mean())
            row[f"{delta_col}_ci_low"] = lo
            row[f"{delta_col}_ci_high"] = hi
            row[f"{delta_col}_prob_candidate_better"] = probability_better
            row[f"{delta_col}_paired_effect_dz"] = (
                orientation * float(train_deltas.mean()) / train_std
                if train_std > 0.0 else float("nan"))
            row[f"{delta_col}_signflip_p"] = paired_sign_flip_p(train_deltas)
        delta_rows.append(row)

    for metric in analysis_metrics:
        key = f"delta_{metric}_signflip_p"
        corrected = holm_adjusted_pvalues([
            float(row.get(key, float("nan"))) for row in delta_rows
        ])
        for row, value in zip(delta_rows, corrected):
            row[f"{key}_holm"] = value

    out_dir.mkdir(parents=True, exist_ok=True)
    per_eval_path = out_dir / "frozen_per_eval.csv"
    summary_path = out_dir / "frozen_summary.csv"
    paired_path = out_dir / "frozen_paired_deltas.csv"
    summary_frame = pd.DataFrame(summary_rows)
    paired_frame = pd.DataFrame(delta_rows)
    if paired_frame.empty:
        paired_frame = pd.DataFrame(columns=[
            "candidate", "reference", "n_pairs"])
    per_eval.to_csv(per_eval_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    paired_frame.to_csv(paired_path, index=False)
    artifacts = {
        "frozen_per_eval.csv": csv_artifact_record(
            per_eval_path,
            per_eval,
            ["config", "train_seed", "eval_seed"],
        ),
        "frozen_summary.csv": csv_artifact_record(
            summary_path,
            summary_frame,
            ["config"],
        ),
        "frozen_paired_deltas.csv": csv_artifact_record(
            paired_path,
            paired_frame,
            ["candidate", "reference"],
        ),
    }
    (out_dir / "matrix_manifest.json").write_text(json.dumps({
        "manifest_version": MATRIX_MANIFEST_VERSION,
        "model_source_fingerprint_version": (
            MODEL_SOURCE_FINGERPRINT_VERSION),
        "protocol_version": protocol_version,
        "stage": str(stage),
        "independent_confirmation": str(stage) == "confirmation",
        "configs": [config_name(value) for value in configs],
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "train_episodes": int(train_episodes),
        "checkpoint_ep": checkpoint_ep,
        "reference": reference_name,
        "primary_metric": "restricted_total_journey_horizon_min",
        "metrics": analysis_metrics,
        "uncertainty": (
            "crossed bootstrap over training and evaluation seeds; one shared "
            "evaluation-seed resample is used across policies within each draw"
        ),
        "paired_test": (
            "two-sided sign-flip test over train-seed mean deltas; this "
            "conditional training-seed inference is distinct from the crossed "
            "train/evaluation population targeted by bootstrap intervals"
        ),
        "multiple_testing": (
            "Holm family-wise correction across candidate-reference "
            "comparisons, separately for each metric"
        ),
        "strict_complete": True,
        "expected_rollouts": expected_rows,
        "common_random_numbers_verified": True,
        "run_manifests_verified": True,
        "run_source_fingerprint": expected_source,
        "scenario_contract": common_scenario_contract,
        "launch_analysis_sha256": next(iter(launch_analysis_hashes)),
        "analysis_fingerprint": current_analysis,
        "runtime_environment": runtime_environment(),
        "invocation": {
            "argv": [str(value) for value in sys.argv],
            "cwd": str(ROOT),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "artifacts": artifacts,
        "run_manifest_artifacts": run_manifest_artifacts,
        "run_git_provenance": common_run_git,
        "git": aggregate_git,
        "config_fingerprints": {
            config_name(value): config_fingerprint(value)
            for value in configs
        },
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument(
        "--stage",
        choices=sorted(MATRIX_STAGES),
        default="development",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--aggregate-logs-dirs", default=None)
    parser.add_argument("--reference", default=DEFAULT_CONFIGS[0])
    parser.add_argument("--job-start", type=int, default=None)
    parser.add_argument("--job-end", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--suppress-heavy-artifacts", action="store_true")
    args = parser.parse_args()

    configs = parse_csv(args.configs)
    train_seeds = parse_csv(args.train_seeds, int)
    eval_seeds = parse_csv(args.eval_seeds, int)
    if len(configs) != len(set(configs)):
        raise SystemExit("configs must be unique")
    if len(train_seeds) != len(set(train_seeds)):
        raise SystemExit("training seeds must be unique")
    if len(eval_seeds) != len(set(eval_seeds)):
        raise SystemExit("evaluation seeds must be unique")
    if set(train_seeds) & set(eval_seeds):
        raise SystemExit("training and evaluation seeds must be disjoint")
    if config_name(args.reference) not in {
            config_name(value) for value in configs}:
        raise SystemExit("reference config must be included in --configs")
    missing_configs = [
        str(config_path(value)) for value in configs
        if not config_path(value).exists()
    ]
    if missing_configs:
        raise SystemExit(f"missing configs: {missing_configs}")
    logs_dir = Path(args.logs_dir)
    if not args.aggregate_only:
        jobs = selected_jobs(configs, train_seeds, args.job_start, args.job_end)
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [
                pool.submit(
                    run_job,
                    config,
                    train_seed,
                    eval_seeds,
                    args.train_episodes,
                    logs_dir,
                    args.worker_threads,
                    args.suppress_heavy_artifacts,
                    args.clean,
                    args.skip_existing,
                    args.stage,
                )
                for config, train_seed in jobs
            ]
            for future in as_completed(futures):
                name, seed, path = future.result()
                print(f"DONE {name} train_seed={seed}: {path}")

    aggregate_logs_dirs = (
        parse_csv(args.aggregate_logs_dirs, Path)
        if args.aggregate_logs_dirs else [logs_dir]
    )
    if args.aggregate_only or args.job_start is None and args.job_end is None:
        aggregate(
            configs,
            train_seeds,
            eval_seeds,
            aggregate_logs_dirs,
            Path(args.out_dir),
            args.reference,
            args.train_episodes,
            args.stage,
        )


if __name__ == "__main__":
    main()
