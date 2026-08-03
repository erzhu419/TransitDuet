#!/usr/bin/env python3
"""Train protocol-v2 policies and evaluate frozen checkpoints on new scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
METRIC_DIRECTIONS = {
    "service_cost": "min",
    "avg_wait_observed_min": "min",
    "restricted_wait_horizon_min": "min",
    "passenger_unserved_rate": "min",
    "headway_cv": "min",
    "fleet_overshoot": "min",
    "trip_launch_rate": "max",
    "trip_completion_rate": "max",
}
PROTOCOL_VERSION = "freqduet-eval-v2"
RUN_MANIFEST_VERSION = "freqduet-run-manifest-v1"
RUN_MANIFEST_NAME = "protocol_run_manifest.json"
SOURCE_PACKAGE_DIRS = ["env", "frequency", "lower", "upper", "coupling"]
SOURCE_FIXED_FILES = [
    "runner_v3.py",
    "scripts/run_freqduet_protocol_v2_matrix.py",
    "env/config.json",
    "env/data/passenger_OD.xlsx",
    "env/data/route_news.xlsx",
    "env/data/stop_news.xlsx",
    "env/data/time_table.xlsx",
]


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
    required = {
        "protocol_version", "eval_seed", "checkpoint_ep", "policy_digest",
        "scenario_tape_id", *METRICS,
    }
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
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: non-finite metric {metric}")


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
        frame = pd.read_csv(result)
        validate_evaluation_frame(
            frame,
            eval_seeds,
            result,
            checkpoint_ep=checkpoint_ep,
            protocol_version=protocol_version,
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol_version") != str(protocol_version):
            return False
        if [int(seed) for seed in manifest.get("scenario_seeds", [])] \
                != [int(seed) for seed in eval_seeds]:
            return False
        if str(manifest.get("policy_digest")) != str(
                frame["policy_digest"].iloc[0]):
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
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    manifest_path = result.parent / "evaluation_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{result}: missing evaluation manifest")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "protocol_version": str(protocol_version),
        "config_name": config_name(config),
        "training_seed": int(train_seed),
        "scenario_seeds": [int(seed) for seed in eval_seeds],
        "n_episodes": len(eval_seeds),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{manifest_path}: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    if str(manifest.get("policy_digest")) != str(
            frame["policy_digest"].iloc[0]):
        raise ValueError(f"{manifest_path}: policy digest mismatch")


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
    rng = np.random.RandomState(20260803)
    train_seeds = frame["train_seed"].drop_duplicates().to_numpy()
    blocks = [
        frame[frame["train_seed"].eq(seed)][metric].to_numpy(dtype=float)
        for seed in train_seeds
    ]
    block_sizes = {len(block) for block in blocks}
    if len(block_sizes) == 1:
        values = np.stack(blocks)
        n_train, n_eval = values.shape
        sampled_train = rng.randint(
            0, n_train, size=(int(draws), n_train))
        sampled_eval = rng.randint(
            0, n_eval, size=(int(draws), n_train, n_eval))
        sampled = values[sampled_train[:, :, None], sampled_eval]
        return sampled.mean(axis=(1, 2))

    estimates = []
    for _ in range(int(draws)):
        sampled_train = rng.choice(train_seeds, size=len(train_seeds), replace=True)
        train_means = []
        for seed in sampled_train:
            block = frame[
                frame["train_seed"].eq(seed)][metric].to_numpy(dtype=float)
            sampled_eval = rng.choice(block, size=len(block), replace=True)
            train_means.append(float(sampled_eval.mean()))
        estimates.append(float(np.mean(train_means)))
    return np.asarray(estimates, dtype=np.float64)


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
    train_episodes: int
) -> dict[str, object]:
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "protocol_version": protocol_version_for_config(config),
        "config_name": config_name(config),
        "config_fingerprint": config_fingerprint(config),
        "source_fingerprint": source_fingerprint(),
        "train_seed": int(train_seed),
        "train_episodes": int(train_episodes),
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
    return payload


def git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return process.stdout.strip() if process.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD") or "unavailable",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unavailable",
        "tracked_dirty": bool(run(
            "status", "--porcelain", "--untracked-files=no")),
    }


def aggregate(
    configs: list[str], train_seeds: list[int], eval_seeds: list[int],
    logs_dirs: list[Path], out_dir: Path, reference: str
) -> None:
    protocol_versions = {
        protocol_version_for_config(config) for config in configs}
    if len(protocol_versions) != 1:
        raise ValueError(
            "one matrix cannot mix protocol versions: "
            f"{sorted(protocol_versions)}")
    protocol_version = next(iter(protocol_versions))
    records = []
    missing_runs = []
    run_manifests = []
    runs_without_manifest = []
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
                checkpoint_ep=None,
                protocol_version=protocol_version,
            )
            validate_evaluation_manifest(
                result,
                frame,
                config,
                train_seed,
                eval_seeds,
                protocol_version=protocol_version,
            )
            run_manifest_path = result.parent.parent / RUN_MANIFEST_NAME
            if run_manifest_path.exists():
                manifest = validate_run_manifest(
                    run_manifest_path, source=result.parent.parent)
                expected_config = config_fingerprint(config)
                expected_fields = {
                    "protocol_version": protocol_version,
                    "config_name": name,
                    "config_fingerprint": expected_config,
                    "train_seed": int(train_seed),
                    "eval_seeds": [int(seed) for seed in eval_seeds],
                }
                for key, expected_value in expected_fields.items():
                    if manifest.get(key) != expected_value:
                        raise ValueError(
                            f"{run_manifest_path}: {key} does not match "
                            "the aggregation protocol")
                run_manifests.append(manifest)
            else:
                runs_without_manifest.append(str(result.parent.parent))
            frame["config"] = name
            frame["train_seed"] = int(train_seed)
            records.append(frame)
    if missing_runs:
        raise RuntimeError(f"missing frozen evaluations: {missing_runs}")
    if not records:
        raise RuntimeError("no frozen evaluation files found")
    if run_manifests and runs_without_manifest:
        raise RuntimeError(
            "matrix mixes source-fingerprinted and legacy runs without a "
            f"manifest: {runs_without_manifest}")
    run_source_hashes = {
        manifest["source_fingerprint"]["sha256"]
        for manifest in run_manifests
    }
    if len(run_source_hashes) > 1:
        raise RuntimeError(
            "matrix mixes multiple source fingerprints: "
            f"{sorted(run_source_hashes)}")
    per_eval = pd.concat(records, ignore_index=True)
    expected_rows = len(configs) * len(train_seeds) * len(eval_seeds)
    if len(per_eval) != expected_rows:
        raise RuntimeError(
            f"matrix has {len(per_eval)} rows, expected {expected_rows}")
    tape_counts = per_eval.groupby(
        ["train_seed", "eval_seed"])["scenario_tape_id"].nunique()
    if not tape_counts.eq(1).all():
        raise RuntimeError(
            "paired policies did not use identical scenario tapes")
    summary_rows = []
    for name, frame in per_eval.groupby("config", sort=False):
        row = {
            "config": name,
            "n_train_seeds": int(frame["train_seed"].nunique()),
            "n_eval_seeds": int(frame["eval_seed"].nunique()),
            "n_rollouts": int(len(frame)),
        }
        for metric in METRICS:
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
        for metric in METRICS:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    per_eval.to_csv(out_dir / "frozen_per_eval.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "frozen_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out_dir / "frozen_paired_deltas.csv", index=False)
    (out_dir / "matrix_manifest.json").write_text(json.dumps({
        "protocol_version": protocol_version,
        "configs": [config_name(value) for value in configs],
        "train_seeds": train_seeds,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "reference": reference_name,
        "metrics": METRICS,
        "uncertainty": "hierarchical bootstrap over train seed and eval seed",
        "paired_test": "two-sided sign-flip test over train-seed mean deltas",
        "strict_complete": True,
        "expected_rollouts": expected_rows,
        "common_random_numbers_verified": True,
        "run_manifests_verified": bool(run_manifests),
        "run_source_fingerprint": (
            run_manifests[0]["source_fingerprint"]
            if run_manifests else None
        ),
        "git": git_provenance(),
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
        )


if __name__ == "__main__":
    main()
