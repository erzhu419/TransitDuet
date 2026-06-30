#!/usr/bin/env python3
"""Run classical external baselines with the FreqDuet paper metric format."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import random
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
ROUTE_HEADWAY_POLICY_PREFIXES = (
    "route_value_",
    "route_headway_",
    "route_oracle_",
)
DEFAULT_FIXED_HEADWAY_S = 360.0
DEFAULT_DOMAIN_CONFIGS = {
    "terminal": "F_freqduet_terminal_main_hiro",
    "highnoise": "F_freqduet_gen_highnoise_main_hiro",
    "odshift": "F_freqduet_gen_odshift_main_hiro",
    "rushshift": "F_freqduet_gen_rushshift_main_hiro",
}


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def parse_csv_file(path: str | Path, cast=str) -> list:
    items = []
    with open(path, "r") as f:
        for line in f:
            items.extend(parse_csv_list(line, cast=cast))
    return items


def fixed_headway_target_s(variant: str) -> float | None:
    """Return the target seconds for fixed-headway variants.

    ``fixed_headway`` keeps the historical 360 s baseline.  Dynamic names such
    as ``fixed_headway_330`` and ``fixed_headway_h330`` are used for route-day
    counterfactual candidate sweeps.
    """
    text = str(variant).strip()
    if text == "fixed_headway":
        return DEFAULT_FIXED_HEADWAY_S
    for prefix in ("fixed_headway_", "fixed_h"):
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix):]
        if suffix.startswith("h"):
            suffix = suffix[1:]
        try:
            target = float(suffix)
        except ValueError:
            return None
        if target <= 0:
            return None
        return target
    return None


def is_route_headway_policy_variant(variant: str) -> bool:
    text = str(variant).strip()
    return any(text.startswith(prefix) for prefix in ROUTE_HEADWAY_POLICY_PREFIXES)


def is_known_variant(variant: str) -> bool:
    return (
        str(variant) in BASELINE_VARIANTS
        or fixed_headway_target_s(variant) is not None
        or is_route_headway_policy_variant(variant)
    )


def config_path(name: str) -> Path:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return ROOT / "configs_freqduet" / filename


def resolve_under_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def run_dir_for(config: str, variant: str, seed: int, logs_dir: Path) -> Path:
    return logs_dir / f"{config}_{variant}_seed{seed}"


def diagnostics_complete(run_dir: Path, episodes: int) -> bool:
    csv_path = run_dir / "diagnostics.csv"
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if "ep" in df.columns:
        df = df[df["ep"] < 9000]
    return len(df) >= int(episodes)


def apply_worker_threads(worker_threads: int | None) -> None:
    if worker_threads is None:
        return
    n = str(max(1, int(worker_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[key] = n


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


def load_headway_policy(path: str | Path | None) -> dict[tuple[str, str, int | None], float]:
    """Load route-day policy targets keyed by (variant, config, seed-or-None)."""
    if not path:
        return {}
    policy_path = resolve_under_root(path)
    if not policy_path.exists():
        raise SystemExit(f"headway policy csv does not exist: {policy_path}")
    df = pd.read_csv(policy_path)
    if df.empty:
        raise SystemExit(f"headway policy csv is empty: {policy_path}")
    if "config" not in df.columns:
        raise SystemExit("headway policy csv must include a config column")
    target_col = None
    for col in ("target_headway_s", "selected_target_headway_s", "headway_s"):
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        if "selected_method" in df.columns:
            df["target_headway_s"] = df["selected_method"].map(fixed_headway_target_s)
            target_col = "target_headway_s"
        else:
            raise SystemExit(
                "headway policy csv must include target_headway_s or selected_method")
    variant_col = None
    for col in ("policy_variant", "variant", "policy_name"):
        if col in df.columns:
            variant_col = col
            break
    if variant_col is None:
        df["policy_variant"] = "route_value_policy"
        variant_col = "policy_variant"

    policy: dict[tuple[str, str, int | None], float] = {}
    for _, row in df.iterrows():
        variant = str(row[variant_col]).strip()
        config = str(row["config"]).strip()
        if not variant or not config:
            continue
        target = pd.to_numeric(row[target_col], errors="coerce")
        if not np.isfinite(target) or float(target) <= 0:
            continue
        seed_key = None
        if "seed" in df.columns and not pd.isna(row.get("seed")):
            seed_val = pd.to_numeric(row.get("seed"), errors="coerce")
            if np.isfinite(seed_val):
                seed_key = int(seed_val)
        policy[(variant, config, seed_key)] = float(target)
    if not policy:
        raise SystemExit(f"no usable rows in headway policy csv: {policy_path}")
    return policy


def route_policy_target_s(
    policy: dict[tuple[str, str, int | None], float],
    variant: str,
    config: str,
    seed: int,
    default_headway_s: float | None,
) -> float:
    keys = [
        (variant, config, int(seed)),
        (variant, config, None),
        ("*", config, int(seed)),
        ("*", config, None),
    ]
    for key in keys:
        if key in policy:
            return float(policy[key])
    if default_headway_s is not None:
        return float(default_headway_s)
    raise KeyError(
        f"no route headway policy target for variant={variant} config={config} seed={seed}")


def make_env_from_config(config_name: str):
    cfg = load_config(str(config_path(config_name)))
    env_cfg = cfg.get("env", {})
    upper_cfg = cfg.get("upper", {})
    env_path = env_cfg.get("path", "env")
    env = env_bus(
        str(resolve_under_root(env_path)),
        route_sigma=env_cfg.get("route_sigma", cfg.get("env", {}).get("route_sigma", 1.5)),
    )
    env.enable_plot = False
    env._n_fleet_target = upper_cfg.get("N_fleet", 12)
    env.demand_noise = env_cfg.get("demand_noise", 0.0)
    env.demand_scale = env_cfg.get("demand_scale", 1.0)
    env.demand_hourly_multipliers = env_cfg.get("demand_hourly_multipliers", None)
    env.service_start_hour = env_cfg.get("service_start_hour", 6)
    env.service_end_hour = env_cfg.get("service_end_hour", 19)
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


def run_episode_external(
    env,
    variant: str,
    n_fleet: int,
    rng: np.random.RandomState,
    demand_noise: float,
    route_headway_target_s: float | None = None,
):
    env._n_fleet_target = int(n_fleet)
    fixed_target = fixed_headway_target_s(variant)
    if fixed_target is None and route_headway_target_s is not None:
        fixed_target = float(route_headway_target_s)
    base_target = DEFAULT_FIXED_HEADWAY_S if fixed_target is None else fixed_target
    chosen_triple = (base_target, base_target, base_target)
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
        return float(base_target)

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
            if fixed_target is not None:
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


def run_one(
    config: str,
    variant: str,
    seed: int,
    episodes: int,
    logs_dir: Path,
    worker_threads: int | None = None,
    headway_policy: dict[tuple[str, str, int | None], float] | None = None,
    policy_default_headway_s: float | None = None,
) -> Path:
    apply_worker_threads(worker_threads)
    run_dir = run_dir_for(config, variant, seed, logs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env, cfg = make_env_from_config(config)
    upper_cfg = cfg.get("upper", {})
    env_cfg = cfg.get("env", {})
    rng = np.random.RandomState(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    route_headway_target = None
    if is_route_headway_policy_variant(variant):
        route_headway_target = route_policy_target_s(
            headway_policy or {},
            variant=variant,
            config=config,
            seed=int(seed),
            default_headway_s=policy_default_headway_s,
        )
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
            route_headway_target_s=route_headway_target,
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


def selected_jobs(
    configs: list[str],
    variants: list[str],
    seeds: list[int],
    job_start: int | None = None,
    job_end: int | None = None,
) -> list[tuple[str, str, int]]:
    jobs = []
    for config in configs:
        for variant in variants:
            for seed in seeds:
                jobs.append((config, variant, seed))
    total = len(jobs)
    if job_start is None and job_end is None:
        return jobs
    start = 0 if job_start is None else max(0, int(job_start))
    end = total if job_end is None else min(total, int(job_end))
    if end < start:
        end = start
    print(f"Shard jobs [{start},{end}) of {total}")
    return jobs[start:end]


def run_jobs(
    configs: list[str],
    variants: list[str],
    seeds: list[int],
    episodes: int,
    logs_dir: Path,
    workers: int,
    skip_existing: bool = False,
    worker_threads: int | None = None,
    job_start: int | None = None,
    job_end: int | None = None,
    headway_policy: dict[tuple[str, str, int | None], float] | None = None,
    policy_default_headway_s: float | None = None,
) -> None:
    jobs = []
    for config, variant, seed in selected_jobs(
        configs, variants, seeds, job_start=job_start, job_end=job_end
    ):
        run_dir = run_dir_for(config, variant, seed, logs_dir)
        if skip_existing and diagnostics_complete(run_dir, episodes):
            print(
                f"SKIP {config} {variant} seed={seed}: "
                f"diagnostics already has >= {episodes} rows"
            )
            continue
        jobs.append((config, variant, seed))
    if not jobs:
        return

    workers = max(1, int(workers))
    if workers == 1:
        for config, variant, seed in jobs:
            run_dir = run_one(
                config, variant, seed, episodes, logs_dir,
                worker_threads=worker_threads,
                headway_policy=headway_policy,
                policy_default_headway_s=policy_default_headway_s,
            )
            print(f"DONE {config} {variant} seed={seed}: {run_dir}")
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_one, config, variant, seed, episodes, logs_dir,
                worker_threads,
                headway_policy,
                policy_default_headway_s,
            )
            for config, variant, seed in jobs
        ]
        for fut in as_completed(futures):
            run_dir = fut.result()
            print(f"DONE {run_dir.name}: {run_dir}")


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


def aggregate(logs_dirs: list[Path] | Path, out_dir: Path, last_k: int) -> None:
    if isinstance(logs_dirs, (str, Path)):
        logs_dirs = [Path(logs_dirs)]
    rows = []
    for logs_dir in logs_dirs:
        for diag in sorted(Path(logs_dir).glob("*/diagnostics.csv")):
            rows.append(summarize_run(diag.parent, last_k=last_k))
    if not rows:
        roots = ", ".join(str(p) for p in logs_dirs)
        raise SystemExit(f"No diagnostics.csv found under {roots}")
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
            "logs_dirs": [str(p) for p in logs_dirs],
            "last_k": int(last_k),
            "n_rows": int(len(per_seed)),
        }, f, indent=2)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT_DOMAIN_CONFIGS.values()))
    ap.add_argument("--configs-file", default=None,
                    help="file containing comma- or newline-separated config names")
    ap.add_argument("--variants", default="fixed_headway,rule_holding,rule_mpc")
    ap.add_argument("--seeds", default="42,123,456,789,2026")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--last-k", type=int, default=20)
    ap.add_argument("--logs-dir", default="logs_external_baselines")
    ap.add_argument("--aggregate-logs-dirs", default=None)
    ap.add_argument("--out-dir", default="results_freqduet/external_baselines")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="run selected jobs but skip summary writing; useful for scheduler shards")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel baseline processes")
    ap.add_argument("--worker-threads", type=int, default=None,
                    help="numeric-library threads per baseline process")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose diagnostics already has enough rows")
    ap.add_argument("--headway-policy-csv", default=None,
                    help="CSV mapping route policy variants/configs[/seeds] to target headways")
    ap.add_argument("--policy-default-headway", type=float, default=None,
                    help="fallback target for route policy variants missing from the CSV")
    ap.add_argument("--job-start", type=int, default=None,
                    help="flattened config x variant x seed start index for scheduler shards")
    ap.add_argument("--job-end", type=int, default=None,
                    help="flattened config x variant x seed end index for scheduler shards")
    args = ap.parse_args()

    configs = (
        parse_csv_file(args.configs_file)
        if args.configs_file else parse_csv_list(args.configs)
    )
    variants = parse_csv_list(args.variants)
    seeds = parse_csv_list(args.seeds, int)
    unknown = sorted(v for v in set(variants) if not is_known_variant(v))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    if (
        not args.aggregate_only
        and any(is_route_headway_policy_variant(v) for v in variants)
        and not args.headway_policy_csv
    ):
        raise SystemExit(
            "route headway policy variants require --headway-policy-csv")
    headway_policy = load_headway_policy(args.headway_policy_csv)

    logs_dir = resolve_under_root(args.logs_dir)
    out_dir = resolve_under_root(args.out_dir)
    env_job_start = (
        os.environ.get("SCHEDULEURM_CPU_START")
        or os.environ.get("SCHEDULEURM_CPU_SHARD_START")
    )
    env_job_end = (
        os.environ.get("SCHEDULEURM_CPU_END")
        or os.environ.get("SCHEDULEURM_CPU_SHARD_END")
    )
    job_start = args.job_start
    job_end = args.job_end
    if job_start is None and env_job_start is not None:
        job_start = int(env_job_start)
    if job_end is None and env_job_end is not None:
        job_end = int(env_job_end)
    if not args.aggregate_only:
        run_jobs(
            configs=configs,
            variants=variants,
            seeds=seeds,
            episodes=args.episodes,
            logs_dir=logs_dir,
            workers=args.workers,
            skip_existing=args.skip_existing,
            worker_threads=args.worker_threads,
            job_start=job_start,
            job_end=job_end,
            headway_policy=headway_policy,
            policy_default_headway_s=args.policy_default_headway,
        )
    if args.no_aggregate:
        print("Skipped aggregation (--no-aggregate).")
        return

    if args.aggregate_logs_dirs:
        aggregate_logs_dirs = [
            resolve_under_root(p) for p in parse_csv_list(args.aggregate_logs_dirs)
        ]
    else:
        aggregate_logs_dirs = [logs_dir]
    aggregate(aggregate_logs_dirs, out_dir, args.last_k)


if __name__ == "__main__":
    main()
