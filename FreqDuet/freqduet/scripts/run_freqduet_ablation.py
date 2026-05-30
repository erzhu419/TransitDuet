#!/usr/bin/env python3
"""Run and aggregate small FreqDuet frequency-allocation ablations.

This script is intentionally separate from the original TransitDuet paper
pipeline. It compares the new frequency interfaces before we add promotion
gates or timetable-spline actions.

Example:
    python scripts/run_freqduet_ablation.py --episodes 30 --seeds 42,123,456
    python scripts/run_freqduet_ablation.py --episodes 5 --seeds 42 --workers 2
    python scripts/run_freqduet_ablation.py --aggregate-only
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "F_nofreq_hiro",
    "F_rawhistory_hiro",
    "F_freqduet_haar_hiro",
    "F_freqduet_harmonic_hiro",
    "F_freqduet_timetable_hiro",
    "F_allfreq_alllayers_hiro",
    "F_swapped_freq_hiro",
]


def parse_csv_list(value, cast=str):
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def config_path(name):
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return Path("configs_freqduet") / filename


def run_dir_for(config, seed, logs_dir):
    return logs_dir / f"{config}_seed{seed}"


def diagnostics_complete(run_dir, episodes):
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


def worker_env(worker_threads=None):
    env = os.environ.copy()
    if worker_threads is not None:
        n = str(max(1, int(worker_threads)))
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "TORCH_NUM_THREADS",
            "FREQDUET_TORCH_THREADS",
        ]:
            env[key] = n
    return env


def run_one(config, seed, episodes, logs_dir, gpu=False, clean=False,
            upper_warmup_eps=None, worker_threads=None):
    run_dir = run_dir_for(config, seed, logs_dir)
    if clean and run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable,
        "runner_v3.py",
        "--config",
        str(config_path(config)),
        "--episodes",
        str(int(episodes)),
        "--seed",
        str(int(seed)),
        "--no-resume",
        "--logs-dir",
        str(logs_dir),
    ]
    if gpu:
        cmd.append("--gpu")
    if upper_warmup_eps is not None:
        cmd.extend(["--upper-warmup-eps", str(int(upper_warmup_eps))])
    thread_note = "" if worker_threads is None else f" threads={int(worker_threads)}"
    print(f"RUN {config} seed={seed} episodes={episodes}{thread_note}")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=worker_env(worker_threads),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "harness_stdout.log").open("w") as f:
        f.write(proc.stdout or "")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"{config} seed={seed} failed with code {proc.returncode}\n{tail}")
    return config, int(seed), run_dir


def run_jobs(configs, seeds, episodes, logs_dir, workers, gpu=False,
             clean=False, skip_existing=False, upper_warmup_eps=None,
             worker_threads=None):
    jobs = []
    for cfg in configs:
        for seed in seeds:
            run_dir = run_dir_for(cfg, seed, logs_dir)
            if skip_existing and diagnostics_complete(run_dir, episodes):
                print(f"SKIP {cfg} seed={seed}: diagnostics already has >= {episodes} rows")
                continue
            jobs.append((cfg, seed))
    if not jobs:
        return

    workers = max(1, int(workers))
    if workers == 1:
        for cfg, seed in jobs:
            run_one(
                cfg, seed, episodes, logs_dir, gpu=gpu, clean=clean,
                upper_warmup_eps=upper_warmup_eps,
                worker_threads=worker_threads)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_one, cfg, seed, episodes, logs_dir,
                gpu=gpu, clean=clean, upper_warmup_eps=upper_warmup_eps,
                worker_threads=worker_threads)
            for cfg, seed in jobs
        ]
        for fut in as_completed(futures):
            cfg, seed, run_dir = fut.result()
            print(f"DONE {cfg} seed={seed}: {run_dir}")


def composite_row(df):
    wait = df["avg_wait_min"].astype(float)
    cv = df["headway_cv"].astype(float)
    overshoot = (
        df["fleet_overshoot"].astype(float)
        if "fleet_overshoot" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    n_fleet = (
        df["N_fleet"].astype(float).clip(lower=1.0)
        if "N_fleet" in df.columns
        else pd.Series(12.0, index=df.index)
    )
    composite = wait / 10.0 + (overshoot ** 2) / n_fleet + cv
    return {
        "wait": float(wait.mean()),
        "cv": float(cv.mean()),
        "overshoot": float(overshoot.mean()),
        "composite": float(composite.mean()),
    }


def summarize_seed(csv_path, last_k):
    df = pd.read_csv(csv_path)
    if "ep" in df.columns:
        df = df[df["ep"] < 9000]
    if df.empty:
        return None
    tail = df.iloc[-min(int(last_k), len(df)):]
    row = composite_row(tail)
    for col in [
        "upper_hf_power_ratio",
        "lower_lf_drift_ratio",
        "demand_attr_score",
        "lower_action_mean",
        "freq_low_demand",
        "freq_low_forecast",
        "freq_high_energy",
        "freq_od_entropy",
        "freq_promotion_flag",
        "freq_promotion_strength",
        "freq_promotion_age",
        "freq_promotion_score",
        "freq_promotion_absorptions",
        "freq_promotion_absorbed",
        "lower_drift_penalty_mean",
        "upper_hf_penalty_mean",
        "freq_wait_lower_penalty_mean",
        "freq_wait_upper_credit_std",
        "freq_wait_low_share_mean",
        "freq_wait_boarded_pax",
    ]:
        row[col] = float(tail[col].astype(float).mean()) if col in tail.columns else 0.0
    row["episodes"] = int(len(df))
    return row


def aggregate(configs, seeds, last_k, logs_dir, out_dir):
    per_seed = []
    for cfg in configs:
        for seed in seeds:
            run_dir = logs_dir / f"{cfg}_seed{seed}"
            csv_path = run_dir / "diagnostics.csv"
            if not csv_path.exists():
                continue
            row = summarize_seed(csv_path, last_k)
            if row is None:
                continue
            row.update({"config": cfg, "seed": int(seed)})
            per_seed.append(row)

    metrics = [
        "wait",
        "cv",
        "overshoot",
        "composite",
        "upper_hf_power_ratio",
        "lower_lf_drift_ratio",
        "demand_attr_score",
        "lower_action_mean",
        "freq_high_energy",
        "freq_promotion_flag",
        "freq_promotion_strength",
        "freq_promotion_absorptions",
        "freq_promotion_absorbed",
        "lower_drift_penalty_mean",
        "upper_hf_penalty_mean",
        "freq_wait_lower_penalty_mean",
        "freq_wait_upper_credit_std",
        "freq_wait_low_share_mean",
        "freq_wait_boarded_pax",
    ]
    summary = []
    for cfg in configs:
        rows = [r for r in per_seed if r["config"] == cfg]
        if not rows:
            continue
        item = {"config": cfg, "n_seeds": len(rows)}
        for metric in metrics:
            vals = np.asarray([r[metric] for r in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std())
        summary.append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "freqduet_ablation_summary.json").open("w") as f:
        json.dump({"per_seed": per_seed, "summary": summary}, f, indent=2)
    pd.DataFrame(per_seed).to_csv(out_dir / "freqduet_ablation_per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "freqduet_ablation_summary.csv", index=False)
    return summary


def print_summary(summary):
    print("=" * 124)
    print(f"{'config':30s} {'wait':>12s} {'cv':>10s} {'comp':>12s} "
          f"{'U_HF':>10s} {'L_LF':>10s} {'attr':>10s} {'prom':>10s}")
    print("-" * 124)
    for r in summary:
        print(
            f"{r['config']:30s} "
            f"{r['wait_mean']:6.2f}±{r['wait_std']:<5.2f} "
            f"{r['cv_mean']:5.3f}±{r['cv_std']:<5.3f} "
            f"{r['composite_mean']:6.3f}±{r['composite_std']:<5.3f} "
            f"{r['upper_hf_power_ratio_mean']:5.3f}±{r['upper_hf_power_ratio_std']:<5.3f} "
            f"{r['lower_lf_drift_ratio_mean']:5.3f}±{r['lower_lf_drift_ratio_std']:<5.3f} "
            f"{r['demand_attr_score_mean']:5.3f}±{r['demand_attr_score_std']:<5.3f} "
            f"{r.get('freq_promotion_strength_mean', 0.0):5.3f}±"
            f"{r.get('freq_promotion_strength_std', 0.0):<5.3f}"
        )
    print("=" * 124)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="comma-separated config names without .yaml")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--last-k", type=int, default=10)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir", default="results_freqduet/ablation")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel runner processes; keep small on CPU")
    ap.add_argument("--worker-threads", type=int, default=None,
                    help="numeric-library threads per runner; opt-in for CPU profiling/load control")
    ap.add_argument("--clean", action="store_true",
                    help="delete each selected run directory before running")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose diagnostics already has enough rows")
    ap.add_argument("--upper-warmup-eps", type=int, default=None,
                    help="override coupling.upper_warmup_eps, useful for short pilots")
    args = ap.parse_args()

    configs = parse_csv_list(args.configs, str)
    seeds = parse_csv_list(args.seeds, int)
    logs_dir = ROOT / args.logs_dir
    if not args.aggregate_only:
        run_jobs(
            configs=configs,
            seeds=seeds,
            episodes=args.episodes,
            logs_dir=logs_dir,
            workers=args.workers,
            gpu=args.gpu,
            clean=args.clean,
            skip_existing=args.skip_existing,
            upper_warmup_eps=args.upper_warmup_eps,
            worker_threads=args.worker_threads,
        )

    summary = aggregate(
        configs=configs,
        seeds=seeds,
        last_k=args.last_k,
        logs_dir=logs_dir,
        out_dir=ROOT / args.out_dir,
    )
    print_summary(summary)
    print(f"Wrote {ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
