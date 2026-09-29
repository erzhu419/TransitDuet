#!/usr/bin/env python3
"""Train corrected main-table learned baselines and test their final checkpoints.

This scheduler-facing entry point keeps checkpoints on the compute node. Each
completed item emits only one held-out JSON summary through
``heldout_final_eval.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from runner_v3 import TransitDuetV2Runner, load_config
from scripts.heldout_final_eval import CHECKPOINT_EPISODE, LEARNED_METHODS, SEEDS, evaluate_item


def train_and_evaluate(method: str, seed: int, episodes: int, n_test: int,
                       device: str, output_dir: str) -> str:
    torch.set_num_threads(1)
    config = load_config(str(SCRIPT_DIR / LEARNED_METHODS[method]))
    config["seed"] = seed
    runner = TransitDuetV2Runner(config, device=device)
    runner.train(total_episodes=episodes)
    del runner
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    path = evaluate_item(
        method=method,
        seed=seed,
        n_test=n_test,
        checkpoint_episode=CHECKPOINT_EPISODE,
        device=device,
        output_dir=SCRIPT_DIR / output_dir,
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(LEARNED_METHODS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-main", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results_remote/revision3_heldout")
    args = parser.parse_args()

    if args.all_main == (args.method is not None):
        parser.error("provide exactly one of --all-main or --method")
    if args.method is not None and args.seed is None:
        parser.error("--seed is required with --method")
    if args.episodes != CHECKPOINT_EPISODE + 1:
        parser.error(f"--episodes must equal {CHECKPOINT_EPISODE + 1} for the pre-specified final checkpoint")
    if args.workers < 1 or args.n_test < 1:
        parser.error("--workers and --n-test must be positive")

    if args.method is not None:
        items = [(args.method, args.seed)]
    else:
        items = [(method, seed) for method in LEARNED_METHODS for seed in SEEDS]
        end = len(items) if args.end is None else args.end
        if args.start < 0 or end < args.start or end > len(items):
            parser.error(f"invalid item range [{args.start}, {end}) for {len(items)} items")
        items = items[args.start:end]

    worker_count = min(args.workers, len(items))
    if worker_count == 1:
        for method, seed in items:
            print(train_and_evaluate(method, seed, args.episodes, args.n_test,
                                     args.device, args.output_dir), flush=True)
        print("Eval complete", flush=True)
        return

    failures = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(train_and_evaluate, method, seed, args.episodes, args.n_test,
                        args.device, args.output_dir): (method, seed)
            for method, seed in items
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(f"{method} seed{seed}: {exc}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print("Eval complete", flush=True)


if __name__ == "__main__":
    main()
