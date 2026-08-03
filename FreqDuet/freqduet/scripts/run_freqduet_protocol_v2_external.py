#!/usr/bin/env python3
"""Evaluate external rules under the frozen FreqDuet protocol-v2 scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner_v3 import load_config
from run_freqduet_external_baselines import (
    config_path,
    is_known_variant,
    make_env_from_config,
    run_episode_external,
)


PROTOCOL_VERSION = "freqduet-eval-v2"
DEFAULT_CONFIG = "F_freqduet_protocol_v2_main_hiro"
DEFAULT_VARIANTS = ["fixed_headway", "rule_holding", "rule_mpc"]
DEFAULT_EVAL_SEEDS = [
    10001, 10007, 10009, 10037, 10039, 10061,
    10067, 10069, 10079, 10091, 10103, 10111,
]


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def config_name(value: str) -> str:
    return Path(value).stem


def external_policy_digest(
    config: str, variant: str, n_fleet: int, protocol_version: str
) -> str:
    payload = json.dumps({
        "protocol_version": str(protocol_version),
        "config": config_name(config),
        "variant": str(variant),
        "n_fleet": int(n_fleet),
        "fixed_headway_s": 360.0,
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def evaluate_variant(
    config: str,
    variant: str,
    eval_seeds: list[int],
    output_dir: Path,
) -> list[dict]:
    env, cfg = make_env_from_config(config)
    env._freqduet_training = False
    upper = cfg.get("upper", {}) or {}
    objective = cfg.get("objective", {}) or {}
    protocol_version = str(
        (cfg.get("protocol", {}) or {}).get(
            "version", PROTOCOL_VERSION))
    n_fleet = int(upper.get("N_fleet", 12))
    digest = external_policy_digest(
        config, variant, n_fleet, protocol_version)
    rows = []
    for eval_seed in eval_seeds:
        random.seed(int(eval_seed))
        np.random.seed(int(eval_seed))
        rng = np.random.RandomState(int(eval_seed))
        env.scenario_seed = int(eval_seed)
        row = run_episode_external(
            env,
            variant=variant,
            n_fleet=n_fleet,
            rng=rng,
            demand_noise=float((cfg.get("env", {}) or {}).get(
                "demand_noise", 0.0)),
            objective_wait_metric=str(
                objective.get("wait_metric", "observed")),
            objective_weights=dict(objective.get("weights", {}) or {}),
        )
        row = dict(row)
        row.update({
            "protocol_version": protocol_version,
            "config": config_name(config),
            "variant": variant,
            "eval_seed": int(eval_seed),
            "policy_digest": digest,
            "service_cost": float(row.pop("composite")),
        })
        rows.append(row)
        print(
            f"{variant} seed={eval_seed} N={n_fleet} "
            f"cost={row['service_cost']:.4f} "
            f"wait={row['avg_wait_observed_min']:.3f} "
            f"cv={row['headway_cv']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{variant}_evaluation.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{variant}_manifest.json").write_text(json.dumps({
        "protocol_version": protocol_version,
        "config_name": config_name(config),
        "variant": variant,
        "fleet_protocol": "fixed_config_default",
        "N_fleet": n_fleet,
        "scenario_seeds": [int(seed) for seed in eval_seeds],
        "policy_digest": digest,
        "n_episodes": len(rows),
    }, indent=2) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    variants = parse_csv(args.variants)
    eval_seeds = parse_csv(args.eval_seeds, int)
    if len(eval_seeds) != len(set(eval_seeds)):
        raise SystemExit("evaluation seeds must be unique")
    unknown = sorted(variant for variant in variants
                     if not is_known_variant(variant))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    resolved = load_config(str(config_path(args.config)))
    protocol_version = str(
        (resolved.get("protocol", {}) or {}).get("version", ""))
    if not protocol_version.startswith("freqduet-eval-v"):
        raise SystemExit(
            "external frozen evaluation requires a versioned protocol config")

    output_dir = Path(args.out_dir)
    all_rows = []
    for variant in variants:
        all_rows.extend(evaluate_variant(
            args.config, variant, eval_seeds, output_dir))
    with (output_dir / "external_evaluation.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
