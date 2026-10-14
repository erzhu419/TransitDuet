#!/usr/bin/env python3
"""Compare optimized two-episode training to the archived original V34 smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from run_freqduet_protocol_v6_checkpoint_trajectory import (
    CURRENT, assert_same_deployment, assert_same_state, runtime_config, seed_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--original-smoke", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root))
    import torch
    from runner_v3 import TransitDuetV2Runner, load_config

    os.environ.pop("FREQDUET_SUPPRESS_HEAVY_ARTIFACTS", None)
    torch.set_num_threads(1)
    source_config = load_config(str(args.model_root / "configs_freqduet" / f"{CURRENT}.yaml"))
    seed_runtime(torch, 12011)
    runner = TransitDuetV2Runner(
        runtime_config(source_config, 12011, args.out_root / "intermediate", 1), device="cpu")
    started = time.perf_counter()
    runner.train(total_episodes=2)
    elapsed = time.perf_counter() - started
    original = args.original_smoke / "intermediate" / f"{CURRENT}_seed12011/checkpoints"
    optimized = Path(runner.log_dir) / "checkpoints"
    for episode in (0, 1):
        for component in ("lower", "upper", "runner"):
            before = torch.load(original / f"{component}_ep{episode}.pt",
                                map_location="cpu", weights_only=False)
            after = torch.load(optimized / f"{component}_ep{episode}.pt",
                               map_location="cpu", weights_only=False)
            if component == "runner":
                assert_same_deployment(before, after, (
                    before["config_fingerprint_sha256"], runner.config_fingerprint_sha256))
            else:
                assert_same_state(before, after)
            print(f"EXACT_TRAINING_MATCH episode={episode} component={component}", flush=True)
    result = {"matches_original_smoke": True, "episodes": 2, "elapsed_s": elapsed,
              "components": ["lower", "upper", "operational_deployment"]}
    (args.out_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
