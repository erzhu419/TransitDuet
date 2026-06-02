#!/usr/bin/env python3
"""Run classical external baselines with the FreqDuet paper metric format."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sim import env_bus
from runner_v3 import load_config
from run_baseline_rule import hour_to_slot, mpc_plan, rule_holding_action


BASELINE_VARIANTS = ("fixed_headway", "rule_holding", "rule_mpc")
DEFAULT_DOMAIN_CONFIGS = {
    "terminal": "F_freqduet_terminal_main_hiro",
    "highnoise": "F_freqduet_gen_highnoise_main_hiro",
    "odshift": "F_freqduet_gen_odshift_main_hiro",
    "rushshift": "F_freqduet_gen_rushshift_main_hiro",
}


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def config_path(name: str) -> Path:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return ROOT / "configs_freqduet" / filename


def infer_domain(config: str) -> str:
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "unknown"


def make_env_from_config(config_name: str):
    cfg = load_config(str(config_path(config_name)))
    env_cfg = cfg.get("env", {})
    upper_cfg = cfg.get("upper", {})
    env = env_bus(str(ROOT / "env"), route_sigma=env_cfg.get("route_sigma", cfg.get("env", {}).get("route_sigma", 1.5)))
    env.enable_plot = False
    env._n_fleet_target = upper_cfg.get("N_fleet", 12)
    env.demand_noise = env_cfg.get("demand_noise", 0.0)
    env.demand_scale = env_cfg.get("demand_scale", 1.0)
    env.od_noise = env_cfg.get("od_noise", 0.0)
    env.od_noise_clip = env_cfg.get("od_noise_clip", [0.3, 2.0])
    env.peak_shift_choices = env_cfg.get("peak_shift_choices", None)
    env.peak_shift_probs = env_cfg.get("peak_shift_probs", None)
    return env, cfg


def composite(wait: float, cv: float, overshoot: float, n_fleet: int) -> float:
    return float(wait / 10.0 + (overshoot ** 2) / max(int(n_fleet), 1) + cv)


def mpc_candidates() -> list[tuple[float, float, float]]:
    triples = []
    for hp in [240, 300, 360, 420, 480]:
        for ho in [360, 480, 600, 720]:
            for ht in [300, 360, 420]:
                triples.append((float(hp), float(ho), float(ht)))
    return triples


def run_episode_external(env, variant: str, n_fleet: int, rng: np.random.RandomState, demand_noise: float):
    env._n_fleet_target = int(n_fleet)
    chosen_triple = (360.0, 360.0, 360.0)
    candidates = None
    if variant == "rule_mpc":
        candidates = mpc_candidates()
        demand_proxy = float(np.clip(rng.normal(1.0, max(float(demand_noise), 1e-9)), 0.3, 2.0))

    def upper_cb(s_upper, trip):
        nonlocal chosen_triple
        trip._upper_queried = True
        if variant == "rule_mpc":
            hour = int(6 + trip.launch_time // 3600)
            chosen_triple = mpc_plan(
                current_hour=hour,
                last_dispatch_time_per_dir=None,
                episode_budget_n=n_fleet,
                current_demand_proxy=demand_proxy,
                candidates=candidates,
            )
            return float(chosen_triple[hour_to_slot(hour)])
        return 360.0

    env.reset()
    env._upper_policy_callback = upper_cb
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: 0.0 for k in range(env.max_agent_num)}
    last_target = {k: 360.0 for k in range(env.max_agent_num)}

    while not env.done:
        for key in state_dict:
            obs_list = state_dict[key]
            if not obs_list:
                continue
            obs = np.array(obs_list[0], dtype=np.float32)
            for bus in env.bus_all:
                if bus.bus_id == int(obs[0]) and bus.on_route:
                    last_target[key] = float(getattr(bus, "_target_headway", 360.0))
                    break
            if variant == "fixed_headway":
                action_dict[key] = 0.0
            else:
                action_dict[key] = rule_holding_action(obs, last_target[key])
            if len(obs_list) == 2:
                state_dict[key] = state_dict[key][1:]
        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    z = env.measurement_vector
    wait = float(z[0])
    peak_fleet = float(z[1])
    cv = float(z[2])
    overshoot = max(0.0, peak_fleet - float(n_fleet))
    return {
        "avg_wait_min": wait,
        "peak_fleet": peak_fleet,
        "headway_cv": cv,
        "N_fleet": int(n_fleet),
        "fleet_overshoot": overshoot,
        "composite": composite(wait, cv, overshoot, n_fleet),
        "target_peak": chosen_triple[0],
        "target_offpeak": chosen_triple[1],
        "target_transition": chosen_triple[2],
    }


def run_one(config: str, variant: str, seed: int, episodes: int, logs_dir: Path) -> Path:
    run_dir = logs_dir / f"{config}_{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env, cfg = make_env_from_config(config)
    upper_cfg = cfg.get("upper", {})
    env_cfg = cfg.get("env", {})
    rng = np.random.RandomState(int(seed))
    np.random.seed(int(seed))
    rows = []
    t_start = time.time()
    for ep in range(int(episodes)):
        if upper_cfg.get("fleet_mode", "fixed") == "elastic":
            n_fleet = int(rng.randint(int(upper_cfg.get("fleet_min", 8)), int(upper_cfg.get("fleet_max", 16)) + 1))
        else:
            n_fleet = int(upper_cfg.get("N_fleet", 12))
        t0 = time.time()
        row = run_episode_external(
            env,
            variant=variant,
            n_fleet=n_fleet,
            rng=rng,
            demand_noise=float(env_cfg.get("demand_noise", 0.0)),
        )
        row.update({
            "ep": ep,
            "variant": variant,
            "config": config,
            "domain": infer_domain(config),
            "seed": int(seed),
            "wall_s": round(time.time() - t0, 3),
        })
        rows.append(row)
        if ep % 5 == 0 or ep == int(episodes) - 1:
            print(
                f"{variant} {config} seed={seed} ep={ep:03d} "
                f"N={row['N_fleet']:2d} wait={row['avg_wait_min']:.2f} "
                f"cv={row['headway_cv']:.3f} over={row['fleet_overshoot']:.1f} "
                f"comp={row['composite']:.3f}"
            )

    diag_path = run_dir / "diagnostics.csv"
    with diag_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "summary.json").open("w") as f:
        json.dump({
            "config": config,
            "domain": infer_domain(config),
            "variant": variant,
            "seed": int(seed),
            "episodes": int(episodes),
            "wall_s": round(time.time() - t_start, 3),
        }, f, indent=2)
    return run_dir


def summarize_run(run_dir: Path, last_k: int) -> dict:
    df = pd.read_csv(run_dir / "diagnostics.csv")
    tail = df.iloc[-min(int(last_k), len(df)):]
    row = {
        "config": str(tail["config"].iloc[0]),
        "domain": str(tail["domain"].iloc[0]),
        "method": str(tail["variant"].iloc[0]),
        "seed": int(tail["seed"].iloc[0]),
        "episodes": int(len(df)),
        "logs_dir": str(run_dir.parent),
    }
    for out_col, src_col in [
        ("wait", "avg_wait_min"),
        ("cv", "headway_cv"),
        ("overshoot", "fleet_overshoot"),
        ("composite", "composite"),
        ("target_peak", "target_peak"),
        ("target_offpeak", "target_offpeak"),
        ("target_transition", "target_transition"),
    ]:
        row[out_col] = float(pd.to_numeric(tail[src_col], errors="coerce").mean())
    return row


def aggregate(logs_dir: Path, out_dir: Path, last_k: int) -> None:
    rows = []
    for diag in sorted(logs_dir.glob("*/diagnostics.csv")):
        rows.append(summarize_run(diag.parent, last_k=last_k))
    if not rows:
        raise SystemExit(f"No diagnostics.csv found under {logs_dir}")
    per_seed = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "external_baselines_per_seed.csv", index=False)

    summary_rows = []
    for (domain, method), group in per_seed.groupby(["domain", "method"], sort=False):
        row = {"domain": domain, "method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in ["wait", "cv", "overshoot", "composite"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "external_baselines_summary.csv", index=False)
    with (out_dir / "external_baselines_summary.json").open("w") as f:
        json.dump({
            "logs_dir": str(logs_dir),
            "last_k": int(last_k),
            "n_rows": int(len(per_seed)),
        }, f, indent=2)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT_DOMAIN_CONFIGS.values()))
    ap.add_argument("--variants", default="fixed_headway,rule_holding,rule_mpc")
    ap.add_argument("--seeds", default="42,123,456,789,2026")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--last-k", type=int, default=20)
    ap.add_argument("--logs-dir", default="logs_external_baselines")
    ap.add_argument("--out-dir", default="results_freqduet/external_baselines")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    configs = parse_csv_list(args.configs)
    variants = parse_csv_list(args.variants)
    seeds = parse_csv_list(args.seeds, int)
    unknown = sorted(set(variants) - set(BASELINE_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    if not args.aggregate_only:
        for config in configs:
            for variant in variants:
                for seed in seeds:
                    run_one(config, variant, seed, args.episodes, logs_dir)
    aggregate(logs_dir, out_dir, args.last_k)


if __name__ == "__main__":
    main()
