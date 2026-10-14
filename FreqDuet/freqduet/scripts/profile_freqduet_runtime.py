#!/usr/bin/env python3
"""Measure a frozen rollout without touching the experiment's model snapshot."""

from __future__ import annotations

import argparse
import cProfile
import copy
import json
from pathlib import Path
import pstats
import random
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-ep", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root.resolve()))
    import numpy as np
    import pandas as pd
    import torch
    from runner_v3 import TransitDuetV2Runner, load_config

    torch.set_num_threads(1)
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    config = copy.deepcopy(load_config(str(
        args.model_root / "configs_freqduet" / f"{args.config}.yaml")))
    config["seed"] = args.train_seed
    config.setdefault("logging", {})["logs_dir"] = str(args.out_root / "logs")
    runner = TransitDuetV2Runner(config, device="cpu")
    runner.load_checkpoint(args.checkpoint_dir, ep=args.checkpoint_ep,
                           require_deployment_state=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    rows, _ = runner.evaluate([args.eval_seed], output_dir=args.out_root / "evaluation",
                              policy_ep=max(args.checkpoint_ep, runner.upper_warmup))
    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - started
    np.savez(args.out_root / "trajectory.npz", **{
        name: np.asarray(getattr(runner, name)) for name in (
            "_ep_lower_actions", "_ep_lower_rewards", "_ep_upper_deltas",
            "_ep_upper_rewards")})
    summary = {
        "model_root": str(args.model_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_ep": args.checkpoint_ep,
        "config": args.config,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "profiled": args.profile,
        "elapsed_s": elapsed,
        "metrics": {name: rows[0][name] for name in (
            "headway_cv", "restricted_total_journey_horizon_min",
            "holding_vehicle_seconds_per_launched_trip", "lower_action_mean",
            "lower_action_std", "upper_delta_mean", "upper_delta_std")},
    }
    if args.compare_to:
        old = pd.read_csv(args.compare_to / "evaluation/evaluation.csv")
        new = pd.read_csv(args.out_root / "evaluation/evaluation.csv")
        # Logging paths identify the benchmark run, not controller behavior.
        excluded = {"wall_env_s", "wall_train_s", "config_fingerprint_sha256", "policy_digest"}
        columns = [name for name in old if name not in excluded]
        pd.testing.assert_frame_equal(old[columns], new[columns], check_exact=True)
        with np.load(args.compare_to / "trajectory.npz") as before, \
                np.load(args.out_root / "trajectory.npz") as after:
            for name in before.files:
                np.testing.assert_array_equal(before[name], after[name])
        reference = json.loads((args.compare_to / "summary.json").read_text())
        summary["comparison"] = {
            "exact_metrics_and_action_reward_traces": True,
            "metric_columns": len(columns),
            "reference_elapsed_s": reference["elapsed_s"],
            "speedup": reference["elapsed_s"] / elapsed,
        }
    if profiler:
        profiler.dump_stats(str(args.out_root / "runtime.pstats"))
        with (args.out_root / "hotspots.txt").open("w") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumulative").print_stats(45)
            stats.sort_stats("tottime").print_stats(35)
        print((args.out_root / "hotspots.txt").read_text(), flush=True)
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
