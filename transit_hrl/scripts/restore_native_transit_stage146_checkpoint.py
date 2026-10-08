#!/usr/bin/env python3
"""Recover a lost native checkpoint by replaying its recorded training recipe."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_diagnostics_stage147 as diagnosis
from scripts import run_native_transit_routing_stage146 as routing
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays


def verify_training_row(row, source, curve):
    ep = row["ep"]
    if row["passengers_generated"] != source["training_demand_counts"][ep]:
        raise RuntimeError(f"Checkpoint recovery changed demand at episode {ep}")
    if ep in curve and routing.compact_row(row) != curve[ep]:
        raise RuntimeError(f"Checkpoint recovery changed recorded training outcome at episode {ep}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=diagnosis.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=diagnosis.ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    source_path = ROOT / "results" / diagnosis.SOURCE_RUN / "cells" / args.method / f"seed_{args.seed}" / "result.json"
    source = json.loads(source_path.read_text())
    if not (source["passed"] and source["contract"] == routing.contract(False)
            and source["protocol"] == routing.EXPERIMENT_PROTOCOL
            and source["method"] == args.method and source["seed"] == args.seed):
        raise RuntimeError("Recovery requires its recorded Stage146 source cell")
    cfg = routing.configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                            args.method, args.seed, preflight=False)
    cfg["logging"] = {"logs_dir": str(raw_directory(source_path) / "training")}
    runner = TransitDuetV2Runner(cfg, device="cpu")
    initial = model_arrays(runner)
    updates = {"upper": 0, "lower": 0}
    for level in updates:
        trainer = getattr(runner, f"{level}_trainer")
        update = trainer.update
        def counted(*pos, _level=level, _update=update, **kw):
            updates[_level] += 1
            return _update(*pos, **kw)
        trainer.update = counted
    curve = {row["ep"]: row for row in source["training_curve"]}
    for ep in range(routing.contract(False)["train_episodes"]):
        row = runner.run_episode(ep, training=True)
        verify_training_row(row, source, curve)
        if ep % 10 == 0 or ep == 299:
            print(f"RECOVERY {args.method} root={args.seed} ep={ep + 1}/300 "
                  f"matches_recorded_training updates={updates}", flush=True)
    final = model_arrays(runner)
    actor_change = {level: max(float(np.max(np.abs(final[k] - initial[k])))
        for k in final if k.startswith(f"{level}/policy_net/")) for level in updates}
    if updates != source["updates"] or actor_change != source["actor_change_max_abs"]:
        raise RuntimeError("Recovered training differs from recorded updates or actor change")
    runner._save_checkpoint(299)
    write_json(args.output.with_name("checkpoint_recovery.json"), {
        "source_run": diagnosis.SOURCE_RUN, "method": args.method, "seed": args.seed,
        "recorded_training_reproduced": True, "training_episodes": 300,
        "native_steps": 300 * routing.contract(False)["training_clock_s"],
        "updates": updates, "actor_change_max_abs": actor_change,
        "boundary": "matches_recorded_training_observables_not_original_weight_array_comparison",
    })
    del runner, initial, final
    command = [sys.executable, "-u", str(ROOT / "scripts/run_native_transit_diagnostics_stage147.py"),
        "--method", args.method, "--seed", str(args.seed), "--output", str(args.output)]
    if args.preflight:
        command.append("--preflight")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
