#!/usr/bin/env python3
"""Diagnose learning duration using paired, frozen V9 checkpoint trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd


SOURCE_COMMIT = "673355cae10640b2737b6d51d541a9685059c342"
MODEL_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/"
    "FreqDuet-v6-engineering-v9-snapshot/FreqDuet/freqduet"
)
REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"
CURRENT = "F_freqduet_protocol_v6_confirmed_main_hiro"
CONFIGS = (REFERENCE, CONTEXT, CURRENT)
V9_TRAIN_SEEDS = (12011, 12037, 12049, 12071, 12097, 12109, 12143, 12161)
V33_TRAIN_SEEDS = (933013, 933031, 933059, 933083)
V9_EVAL_SEEDS = (45007, 45013, 45053, 45061, 45077, 45119, 45131, 45137)
V33_EVAL_SEEDS = (934011, 934029, 934047, 934071)
TRAIN_SEEDS = (*V9_TRAIN_SEEDS, *V33_TRAIN_SEEDS)
EVAL_SEEDS = (*V9_EVAL_SEEDS, *V33_EVAL_SEEDS)
CHECKPOINTS = (39, 79, 119, 159, 199)
RUN_NAME = "protocol_v6_v34_checkpoint_trajectory"
METRICS = (
    "headway_cv",
    "restricted_total_journey_horizon_min",
    "restricted_wait_horizon_min",
    "restricted_in_vehicle_horizon_min",
    "holding_vehicle_seconds_per_launched_trip",
    "fleet_denied_trip_rate",
    "fleet_readiness_delay_mean_s",
    "lower_action_mean",
    "lower_action_std",
    "upper_delta_mean",
    "upper_delta_std",
    "lower_departure_regularity_evidence_valid_mean",
)
KEYS = ["train_seed", "eval_seed", "checkpoint_ep"]


def locked_runtime(model_root: Path):
    sys.path.insert(0, str(model_root))
    from scripts import run_freqduet_protocol_v2_matrix as matrix
    from runner_v3 import TransitDuetV2Runner, load_config
    import torch

    # Reuse the archived experiment's source contract before any training.
    baseline = model_root / "results_freqduet" / (
        "protocol_v6_confirmed_main_ep200_s8_e8_v9/logs_shards"
    )
    manifest_path = next(baseline.glob("*/*/protocol_run_manifest.json"))
    parent = json.loads(manifest_path.read_text())
    if matrix.source_fingerprint() != parent["source_fingerprint"]:
        raise ValueError("V9 model source differs from the recorded experiment")
    if parent["git"]["commit"] != SOURCE_COMMIT:
        raise ValueError("V9 source commit differs from the locked experiment")
    torch.set_num_threads(1)
    return matrix, TransitDuetV2Runner, load_config, torch


def runtime_config(config: dict, seed: int, logs_dir: Path, frequency: int) -> dict:
    result = copy.deepcopy(config)
    result["seed"] = int(seed)
    result.setdefault("logging", {})["logs_dir"] = str(logs_dir.resolve())
    result.setdefault("training", {}).update(
        save_checkpoints=True, checkpoint_freq=int(frequency), trip_dump_freq=0
    )
    return result


def seed_runtime(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluation_rows(path: Path, eval_seeds, checkpoint: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame["eval_seed"].duplicated().any() or set(frame["eval_seed"]) != set(eval_seeds):
        raise ValueError(f"incomplete or duplicated evaluation seeds: {path}")
    if set(frame["checkpoint_ep"]) != {checkpoint}:
        raise ValueError(f"wrong checkpoint in {path}")
    if set(frame["protocol_version"]) != {"freqduet-eval-v6"}:
        raise ValueError(f"wrong evaluation protocol in {path}")
    if frame["policy_digest"].isna().any() or frame["policy_digest"].nunique() != 1:
        raise ValueError(f"frozen deployment state changed in {path}")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite trajectory metrics in {path}")
    return frame


def run_trajectory(model_root: Path, out_root: Path, config: str, seed: int) -> None:
    matrix, Runner, load_config, torch = locked_runtime(model_root)
    os.environ.pop("FREQDUET_SUPPRESS_HEAVY_ARTIFACTS", None)
    source_config = load_config(str(model_root / "configs_freqduet" / f"{config}.yaml"))
    logs_dir = out_root / "logs"
    run_dir = logs_dir / f"{config}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "stage": "exploratory_diagnostic",
        "source_commit": SOURCE_COMMIT,
        "diagnostic_commit": os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
        "source_config": source_config,
        "config": config,
        "train_seed": seed,
        "eval_seeds": list(EVAL_SEEDS),
        "checkpoint_episodes": list(CHECKPOINTS),
        "train_episodes": 200,
        "instrumentation": {"checkpoint_freq": 40, "trip_dump_freq": 0},
    }
    manifest_path = run_dir / "trajectory_contract.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise ValueError(f"existing trajectory has a different contract: {run_dir}")
    manifest_path.write_text(json.dumps(contract, indent=2) + "\n")
    checkpoints_exist = all(
        (run_dir / "checkpoints" / f"{name}_ep{ep}.pt").is_file()
        for ep in CHECKPOINTS for name in ("runner", "lower", "upper")
    )
    if not (checkpoints_exist and matrix.training_complete(run_dir, 200, "freqduet-eval-v6")):
        seed_runtime(torch, seed)
        runner = Runner(runtime_config(source_config, seed, logs_dir, 40), device="cpu")
        runner.train(total_episodes=200)
        del runner
    for checkpoint in CHECKPOINTS:
        evaluate_checkpoint(model_root, out_root, config, seed, checkpoint,
                            runtime=(matrix, Runner, load_config, torch))
    ready = out_root / "ready" / f"{config}_seed{seed}"
    ready.mkdir(parents=True, exist_ok=True)
    (ready / "ready.json").write_text(json.dumps({
        "config": config, "train_seed": seed, "checkpoints": list(CHECKPOINTS),
        "source_commit": SOURCE_COMMIT, "rollouts": len(CHECKPOINTS) * len(EVAL_SEEDS),
    }) + "\n")
    print("TRAJECTORY_JOB_DONE", flush=True)


def evaluate_checkpoint(model_root: Path, out_root: Path, config: str, seed: int,
                        checkpoint: int, runtime=None) -> None:
    """Precompute an available checkpoint while its training worker continues.

    Run only before the owning worker enters evaluation; each checkpoint has
    one producer. Its original worker reuses the same validated output later.
    """
    _, Runner, load_config, torch = runtime or locked_runtime(model_root)
    logs_dir = out_root / "logs"
    run_dir = logs_dir / f"{config}_seed{seed}"
    output = run_dir / "frozen_trajectory" / f"ep{checkpoint}"
    if not (output / "evaluation.csv").is_file():
        source_config = load_config(str(model_root / "configs_freqduet" / f"{config}.yaml"))
        seed_runtime(torch, seed)
        runner = Runner(runtime_config(source_config, seed, logs_dir, 40), device="cpu")
        runner.load_checkpoint(run_dir / "checkpoints", ep=checkpoint, require_deployment_state=True)
        runner.evaluate(EVAL_SEEDS, output_dir=output, policy_ep=max(checkpoint, runner.upper_warmup))
        del runner
    frame = evaluation_rows(output / "evaluation.csv", EVAL_SEEDS, checkpoint)
    print(f"TRAJECTORY_CHECKPOINT_DONE config={config} seed={seed} ep={checkpoint}", flush=True)
    small = frame[["eval_seed", "checkpoint_ep", "scenario_tape_id", *METRICS]].copy()
    small.insert(0, "train_seed", seed)
    small.insert(0, "config", config)
    destination = out_root / "records" / f"{config}_seed{seed}_ep{checkpoint}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    small.to_csv(destination, index=False)


def paired_trajectory(frame: pd.DataFrame, candidate: str, reference: str) -> pd.DataFrame:
    left = frame.loc[frame.config == candidate, [*KEYS, *METRICS]]
    right = frame.loc[frame.config == reference, [*KEYS, *METRICS]]
    paired = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True,
                        suffixes=("_candidate", "_reference"))
    if not paired["_merge"].eq("both").all():
        raise ValueError("trajectory comparison has unmatched policy rows")
    result = paired[KEYS].copy()
    for metric in METRICS:
        result[metric] = paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
    return result


def checkpoint_change(paired: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    keys = ["train_seed", "eval_seed"]
    early = paired.loc[paired.checkpoint_ep == first, [*keys, *METRICS]]
    late = paired.loc[paired.checkpoint_ep == last, [*keys, *METRICS]]
    joined = late.merge(early, on=keys, how="outer", validate="one_to_one", indicator=True,
                        suffixes=("_late", "_early"))
    if not joined["_merge"].eq("both").all():
        raise ValueError("checkpoint change has unmatched early and late rows")
    result = joined[keys].copy()
    for metric in METRICS:
        result[metric] = joined[f"{metric}_late"] - joined[f"{metric}_early"]
    return result


def validate_grid(frame: pd.DataFrame, configs, train_seeds, eval_seeds, checkpoints) -> None:
    keys = ["config", *KEYS]
    expected = pd.MultiIndex.from_product(
        [configs, train_seeds, eval_seeds, checkpoints], names=keys
    )
    observed = pd.MultiIndex.from_frame(frame[keys])
    if observed.has_duplicates or len(expected.difference(observed)) or len(observed.difference(expected)):
        raise ValueError("trajectory grid is incomplete, duplicated, or outside the design")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("trajectory grid has non-finite metrics")
    if not frame.groupby("eval_seed")["scenario_tape_id"].nunique().eq(1).all():
        raise ValueError("scenario tapes differ across policies or checkpoints")


def historical_endpoint(frame: pd.DataFrame, model_root: Path) -> dict:
    root = model_root / "results_freqduet/protocol_v6_confirmed_main_ep200_s8_e8_v9/logs_shards"
    comparisons = []
    for config in CONFIGS:
        for seed in V9_TRAIN_SEEDS:
            paths = list(root.glob(f"*/{config}_seed{seed}/frozen_evaluation/evaluation.csv"))
            if len(paths) != 1:
                raise ValueError(f"expected one archived V9 evaluation for {config}, {seed}")
            old = evaluation_rows(paths[0], V9_EVAL_SEEDS, 199).set_index("eval_seed")
            new = frame.loc[(frame.config == config) & (frame.train_seed == seed)
                            & (frame.checkpoint_ep == 199) & frame.eval_seed.isin(V9_EVAL_SEEDS)]
            new = new.set_index("eval_seed").reindex(old.index)
            differences = (new[list(METRICS)] - old[list(METRICS)]).abs().max()
            comparisons.append({"config": config, "train_seed": seed,
                                "metric_max_abs_difference": differences.to_dict(),
                                "matches": bool(differences.le(1e-9).all())})
    return {"matches_archived_v9": all(r["matches"] for r in comparisons), "runs": comparisons}


def aggregate(out_root: Path, model_root: Path) -> None:
    sys.path.insert(0, str(model_root))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval

    paths = sorted((out_root / "records").glob("*.csv"))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    validate_grid(frame, CONFIGS, TRAIN_SEEDS, EVAL_SEEDS, CHECKPOINTS)
    cohorts = {
        "v9_train_v9_eval": (V9_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v33_train_v9_eval": (V33_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v9_train_v33_eval": (V9_TRAIN_SEEDS, V33_EVAL_SEEDS),
        "v33_train_v33_eval": (V33_TRAIN_SEEDS, V33_EVAL_SEEDS),
        "all_train_all_eval": (TRAIN_SEEDS, EVAL_SEEDS),
    }
    summaries, changes = [], []
    for reference in (REFERENCE, CONTEXT):
        paired = paired_trajectory(frame, CURRENT, reference)
        for cohort, (train_seeds, eval_seeds) in cohorts.items():
            subset = paired.loc[paired.train_seed.isin(train_seeds) & paired.eval_seed.isin(eval_seeds)]
            drift = checkpoint_change(subset, CHECKPOINTS[0], CHECKPOINTS[-1])
            for checkpoint, rows in subset.groupby("checkpoint_ep"):
                for metric in METRICS:
                    low, high = hierarchical_interval(rows, metric)
                    summaries.append({"cohort": cohort, "reference": reference,
                                      "checkpoint_ep": int(checkpoint), "metric": metric,
                                      "delta": float(rows[metric].mean()), "ci95_low": low,
                                      "ci95_high": high,
                                      "negative_train_seed_fraction": float(
                                          rows.groupby("train_seed")[metric].mean().lt(0).mean())})
            for metric in METRICS:
                low, high = hierarchical_interval(drift, metric)
                changes.append({"cohort": cohort, "reference": reference, "metric": metric,
                                "delta_ep199_minus_ep39": float(drift[metric].mean()),
                                "ci95_low": low, "ci95_high": high})
    destination = out_root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "checkpoint_deltas.csv", index=False)
    pd.DataFrame(changes).to_csv(destination / "late_minus_early.csv", index=False)
    endpoint = historical_endpoint(frame, model_root)
    (destination / "historical_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
    report = {"status": "complete_exploratory_diagnostic", "source_commit": SOURCE_COMMIT,
              "rollouts": len(frame), "training_runs": len(CONFIGS) * len(TRAIN_SEEDS),
              "checkpoints": list(CHECKPOINTS), "independent_confirmation": False,
              "matches_archived_v9": endpoint["matches_archived_v9"]}
    (destination / "diagnostic_status.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def assert_same_state(left, right, label="state") -> None:
    import torch

    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"checkpoint keys differ: {label}")
        for key in left:
            assert_same_state(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"checkpoint lengths differ: {label}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same_state(a, b, f"{label}[{index}]")
    elif isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"checkpoint tensor differs: {label}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"checkpoint array differs: {label}")
    elif left != right:
        raise AssertionError(f"checkpoint value differs: {label}")


def assert_same_deployment(left: dict, right: dict, expected_fingerprints) -> None:
    states = []
    for state, expected in zip((left, right), expected_fingerprints):
        if state["config_fingerprint_sha256"] != expected:
            raise AssertionError("checkpoint does not match its instrumentation configuration")
        deployment = copy.deepcopy(state)
        deployment.pop("config_fingerprint_sha256")
        previous = deployment.get("fixed_selector_prev_diag")
        if isinstance(previous, dict):
            if previous["config_fingerprint_sha256"] != expected:
                raise AssertionError("cached diagnostics do not match their instrumentation configuration")
            # Keep operational cached outcomes in the comparison; omit log metadata.
            for key in ("config_fingerprint_sha256", "wall_env_s", "wall_train_s"):
                previous.pop(key, None)
        states.append(deployment)
    assert_same_state(*states)


def verify_smoke(model_root: Path, out_root: Path) -> None:
    _, Runner, load_config, torch = locked_runtime(model_root)
    os.environ.pop("FREQDUET_SUPPRESS_HEAVY_ARTIFACTS", None)
    config = load_config(str(model_root / "configs_freqduet" / f"{CURRENT}.yaml"))
    seed = V9_TRAIN_SEEDS[0]
    expected_fingerprints = []
    for label, frequency in (("intermediate", 1), ("final_only", 0)):
        seed_runtime(torch, seed)
        runner = Runner(runtime_config(config, seed, out_root / label, frequency), device="cpu")
        expected_fingerprints.append(runner.config_fingerprint_sha256)
        del runner
    for component in ("lower", "upper", "runner"):
        files = [out_root / label / f"{CURRENT}_seed{seed}/checkpoints/{component}_ep1.pt"
                 for label in ("intermediate", "final_only")]
        states = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
        if component == "runner":
            assert_same_deployment(*states, expected_fingerprints)
        else:
            assert_same_state(*states)
        print(f"CHECKPOINT_SAVE_EQUIVALENT component={component}", flush=True)
    for checkpoint in (0, 1):
        seed_runtime(torch, seed)
        runner = Runner(runtime_config(config, seed, out_root / "intermediate", 1), device="cpu")
        runner.load_checkpoint(Path(runner.log_dir) / "checkpoints", ep=checkpoint,
                               require_deployment_state=True)
        output = out_root / f"smoke_eval_ep{checkpoint}"
        runner.evaluate(V9_EVAL_SEEDS[:1], output_dir=output,
                        policy_ep=max(checkpoint, runner.upper_warmup))
        evaluation_rows(output / "evaluation.csv", V9_EVAL_SEEDS[:1], checkpoint)
        del runner
    (out_root / "smoke_pass.json").write_text(json.dumps({
        "source_commit": SOURCE_COMMIT, "extra_checkpoint_preserves_state": True,
        "intermediate_frozen_evaluation_passes": True,
    }, indent=2) + "\n")
    print("TRAJECTORY_SMOKE_PASS", flush=True)


def smoke(model_root: Path, out_root: Path) -> None:
    _, Runner, load_config, torch = locked_runtime(model_root)
    os.environ.pop("FREQDUET_SUPPRESS_HEAVY_ARTIFACTS", None)
    config = load_config(str(model_root / "configs_freqduet" / f"{CURRENT}.yaml"))
    seed = V9_TRAIN_SEEDS[0]
    for label, frequency in (("intermediate", 1), ("final_only", 0)):
        seed_runtime(torch, seed)
        runner = Runner(runtime_config(config, seed, out_root / label, frequency), device="cpu")
        runner.train(total_episodes=2)
        del runner
    verify_smoke(model_root, out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "verify-smoke", "aggregate", "evaluate-checkpoint"))
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--checkpoint-ep", type=int, choices=CHECKPOINTS)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if args.command in ("run", "evaluate-checkpoint"):
        if args.config is None or args.train_seed is None:
            parser.error(f"{args.command} requires --config and --train-seed")
        if args.command == "evaluate-checkpoint":
            if args.checkpoint_ep is None:
                parser.error("evaluate-checkpoint requires --checkpoint-ep")
            evaluate_checkpoint(args.model_root, args.out_root, args.config,
                                args.train_seed, args.checkpoint_ep)
        else:
            run_trajectory(args.model_root, args.out_root, args.config, args.train_seed)
    elif args.command == "smoke":
        smoke(args.model_root, args.out_root)
    elif args.command == "verify-smoke":
        verify_smoke(args.model_root, args.out_root)
    else:
        aggregate(args.out_root, args.model_root)


if __name__ == "__main__":
    main()
