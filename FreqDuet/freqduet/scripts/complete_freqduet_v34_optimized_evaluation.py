#!/usr/bin/env python3
"""Complete V34 frozen evaluations using its verified runtime-only optimization."""

import argparse
import json
from pathlib import Path
import sys

from run_freqduet_protocol_v6_checkpoint_trajectory import (
    CHECKPOINTS, CONFIGS, EVAL_SEEDS, MODEL_ROOT as ORIGINAL_ROOT,
    SOURCE_COMMIT, TRAIN_SEEDS, aggregate, evaluate_checkpoint, evaluation_rows,
)


OPTIMIZED_ROOT = ORIGINAL_ROOT.parents[2] / (
    "FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet")
OPTIMIZATION_COMMIT = "9a59ed3b0d"


def optimized_runtime(out_root):
    proof_root = out_root.parent / "runtime_optimization_20261014"
    benchmark = json.loads((proof_root / "benchmark_summary.json").read_text())
    training = json.loads((proof_root / "training_equivalence/summary.json").read_text())
    if not benchmark["all_exact"] or not training["matches_original_smoke"]:
        raise ValueError("runtime optimization has not passed equivalence tests")
    sys.path.insert(0, str(OPTIMIZED_ROOT))
    import torch
    from runner_v3 import TransitDuetV2Runner, load_config
    from scripts import run_freqduet_protocol_v2_matrix as matrix

    torch.set_num_threads(1)
    return matrix, TransitDuetV2Runner, load_config, torch


def audit(out_root):
    runtime = optimized_runtime(out_root)
    matrix, _, _, _ = runtime
    missing, completed = [], 0
    for config in CONFIGS:
        for seed in TRAIN_SEEDS:
            run_dir = out_root / "logs" / f"{config}_seed{seed}"
            if not matrix.training_complete(run_dir, 200, "freqduet-eval-v6"):
                raise ValueError(f"training is not complete; refusing to retrain: {run_dir}")
            contract = json.loads((run_dir / "trajectory_contract.json").read_text())
            if (contract["source_commit"], contract["config"], contract["train_seed"]) != (
                    SOURCE_COMMIT, config, seed):
                raise ValueError(f"unexpected training contract: {run_dir}")
            for checkpoint in CHECKPOINTS:
                for component in ("runner", "lower", "upper"):
                    if not (run_dir / "checkpoints" / f"{component}_ep{checkpoint}.pt").is_file():
                        raise FileNotFoundError(f"missing {component} checkpoint: {run_dir}, {checkpoint}")
                csv = run_dir / "frozen_trajectory" / f"ep{checkpoint}/evaluation.csv"
                if csv.is_file():
                    evaluation_rows(csv, EVAL_SEEDS, checkpoint)
                    record = out_root / "records" / f"{config}_seed{seed}_ep{checkpoint}.csv"
                    if not record.is_file():
                        evaluate_checkpoint(ORIGINAL_ROOT, out_root, config, seed,
                                            checkpoint, runtime=runtime)
                    completed += 1
                else:
                    missing.append(dict(config=config, train_seed=seed, checkpoint_ep=checkpoint))
    result = dict(training_runs=36, train_episodes=200, completed_checkpoints=completed,
                  missing=missing, training_source_commit=SOURCE_COMMIT,
                  runtime_optimization_commit=OPTIMIZATION_COMMIT)
    target = out_root / "runtime_relaunch"
    target.mkdir(parents=True, exist_ok=True)
    (target / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--checkpoint-ep", type=int, choices=CHECKPOINTS)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.out_root)
    elif args.command == "aggregate":
        aggregate(args.out_root, ORIGINAL_ROOT)
        previous = json.loads((args.out_root / "runtime_relaunch/audit.json").read_text())
        provenance = dict(training_source_commit=SOURCE_COMMIT,
                          runtime_optimization_commit=OPTIMIZATION_COMMIT,
                          reused_training_runs=previous["training_runs"],
                          reused_evaluation_checkpoints=previous["completed_checkpoints"],
                          optimized_evaluation_checkpoints=len(previous["missing"]))
        (args.out_root / "summary/evaluation_runtime.json").write_text(
            json.dumps(provenance, indent=2) + "\n")
        for config in CONFIGS:
            for seed in TRAIN_SEEDS:
                ready = args.out_root / "ready" / f"{config}_seed{seed}"
                if not (ready / "ready.json").is_file():
                    ready.mkdir(parents=True, exist_ok=True)
                    (ready / "ready.json").write_text(json.dumps(dict(
                        config=config, train_seed=seed, checkpoints=list(CHECKPOINTS),
                        source_commit=SOURCE_COMMIT, rollouts=60,
                        evaluation_optimization_commit=OPTIMIZATION_COMMIT)) + "\n")
    else:
        if args.config is None or args.train_seed is None or args.checkpoint_ep is None:
            parser.error("evaluate requires --config, --train-seed, --checkpoint-ep")
        runtime = optimized_runtime(args.out_root)
        evaluate_checkpoint(ORIGINAL_ROOT, args.out_root, args.config,
                            args.train_seed, args.checkpoint_ep, runtime=runtime)
        name = f"{args.config}_seed{args.train_seed}_ep{args.checkpoint_ep}"
        ready = args.out_root / "runtime_relaunch/ready" / name
        ready.mkdir(parents=True, exist_ok=True)
        (ready / "ready.json").write_text(json.dumps(dict(
            config=args.config, train_seed=args.train_seed, checkpoint_ep=args.checkpoint_ep,
            training_source_commit=SOURCE_COMMIT,
            runtime_optimization_commit=OPTIMIZATION_COMMIT)) + "\n")


if __name__ == "__main__":
    main()
