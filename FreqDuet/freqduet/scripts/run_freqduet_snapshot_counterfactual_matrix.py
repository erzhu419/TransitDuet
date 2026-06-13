#!/usr/bin/env python3
"""Run and aggregate FreqDuet snapshot counterfactual audits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "F_freqduet_terminal_main_hiro",
    "F_freqduet_gen_highnoise_main_hiro",
    "F_freqduet_gen_odshift_main_hiro",
    "F_freqduet_gen_rushshift_main_hiro",
]
DEFAULT_SEEDS = [
    7, 11, 17, 23, 31, 37, 42, 43, 53, 61,
    71, 83, 97, 109, 123, 127, 149, 456, 789, 2026,
]


def parse_csv(value: str, cast=str) -> list:
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def job_list(configs: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    return [(config, int(seed)) for config in configs for seed in seeds]


def selected_jobs(
    configs: list[str],
    seeds: list[int],
    job_start: int | None,
    job_end: int | None,
) -> list[tuple[str, int]]:
    jobs = job_list(configs, seeds)
    if job_start is None and job_end is None:
        return jobs
    start = 0 if job_start is None else max(0, int(job_start))
    end = len(jobs) if job_end is None else min(len(jobs), int(job_end))
    print(f"Shard jobs [{start},{end}) of {len(jobs)}")
    return jobs[start:end]


def run_dir_for(
    out_dir: Path,
    config: str,
    seed: int,
    episodes: int,
    snapshots: int,
    horizon_s: float,
    burn_in_episodes: int = 0,
) -> Path:
    burn_tag = f"_burn{int(burn_in_episodes)}" if int(burn_in_episodes) > 0 else ""
    return out_dir / (
        f"{Path(config).stem}_seed{int(seed)}_ep{int(episodes)}"
        f"{burn_tag}_snap{int(snapshots)}_h{int(horizon_s)}"
    )


def audit_complete(run_dir: Path, expected_rows: int) -> bool:
    meta_path = run_dir / "snapshot_counterfactual_meta.json"
    csv_path = run_dir / "snapshot_counterfactual_labels.csv"
    if not meta_path.exists() or not csv_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if int(meta.get("rows", 0)) < int(expected_rows):
            return False
        df = pd.read_csv(csv_path, nrows=5)
        return not df.empty
    except Exception:
        return False


def worker_env(worker_threads: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if worker_threads is None:
        return env
    n = str(max(1, int(worker_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "FREQDUET_TORCH_THREADS",
    ):
        env[key] = n
    return env


def run_one(config: str, seed: int, args: argparse.Namespace) -> tuple[str, int, Path]:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    modes = parse_csv(args.modes, str)
    deltas = parse_csv(args.deltas_s, float)
    expected_rows = int(args.max_snapshots) * len(modes) * len(deltas)
    run_dir = run_dir_for(
        out_dir,
        config,
        seed,
        args.episodes,
        args.max_snapshots,
        args.horizon_s,
        args.burn_in_episodes,
    )
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.skip_existing and audit_complete(run_dir, expected_rows):
        print(f"SKIP {config} seed={seed}: existing snapshot labels complete")
        return config, int(seed), run_dir

    cmd = [
        sys.executable,
        "scripts/audit_freqduet_snapshot_counterfactual.py",
        "--config", config,
        "--seed", str(int(seed)),
        "--episodes", str(int(args.episodes)),
        "--burn-in-episodes", str(int(args.burn_in_episodes)),
        "--upper-warmup-eps", str(int(args.upper_warmup_eps)),
        "--max-snapshots", str(int(args.max_snapshots)),
        "--start-dispatch", str(int(args.start_dispatch)),
        "--snapshot-stride", str(int(args.snapshot_stride)),
        "--horizon-s", str(float(args.horizon_s)),
        f"--deltas-s={args.deltas_s}",
        "--modes", args.modes,
        "--terminal-hold-s", str(float(args.terminal_hold_s)),
        "--terminal-min-s", str(float(args.terminal_min_s)),
        "--terminal-floor-ratio", str(float(args.terminal_floor_ratio)),
        "--terminal-floor-min-s", str(float(args.terminal_floor_min_s)),
        "--worker-threads", str(int(args.worker_threads)),
        "--out-dir", str(out_dir),
    ]
    if args.stochastic_lower:
        cmd.append("--stochastic-lower")
    print(f"RUN {config} seed={seed}")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=worker_env(args.worker_threads),
    )
    log_dir = out_dir / "_matrix_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{Path(config).stem}_seed{int(seed)}.log").write_text(proc.stdout or "")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-80:])
        raise RuntimeError(f"{config} seed={seed} failed with code {proc.returncode}\n{tail}")
    return config, int(seed), run_dir


def run_matrix(args: argparse.Namespace, configs: list[str], seeds: list[int]) -> None:
    jobs = selected_jobs(configs, seeds, args.job_start, args.job_end)
    if not jobs:
        return
    if int(args.workers) <= 1:
        for config, seed in jobs:
            run_one(config, seed, args)
        return
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(run_one, config, seed, args) for config, seed in jobs]
        for fut in as_completed(futures):
            config, seed, run_dir = fut.result()
            print(f"DONE {config} seed={seed}: {run_dir}")


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def read_label_files(out_dir: Path) -> pd.DataFrame:
    paths = sorted(out_dir.rglob("snapshot_counterfactual_labels.csv"))
    parts = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        meta_path = path.with_name("snapshot_counterfactual_meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        if "config" not in df.columns and meta.get("config"):
            df["config"] = Path(str(meta["config"])).stem
        if "seed" not in df.columns and meta.get("seed") is not None:
            df["seed"] = int(meta["seed"])
        if "domain" not in df.columns:
            config = str(df["config"].iloc[0]) if "config" in df.columns and len(df) else ""
            if "gen_highnoise" in config:
                domain = "highnoise"
            elif "gen_odshift" in config:
                domain = "odshift"
            elif "gen_rushshift" in config:
                domain = "rushshift"
            elif "terminal" in config:
                domain = "terminal"
            else:
                domain = "unknown"
            df["domain"] = domain
        parts.append(df)
    if not parts:
        raise SystemExit(f"no snapshot_counterfactual_labels.csv under {out_dir}")
    return pd.concat(parts, ignore_index=True)


def summarize_labels(
    df: pd.DataFrame,
    n_boot: int,
    baseline_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["domain", "config", "seed", "ep", "dispatch_index"]
    df = df.copy()
    df["proxy_cost"] = pd.to_numeric(df["proxy_cost"], errors="coerce")
    df = df[np.isfinite(df["proxy_cost"])].copy()

    group_rows = []
    for (domain, method), group in df.groupby(["domain", "candidate_method"], sort=True):
        values = group["proxy_cost"].to_numpy(dtype=np.float64)
        lo, hi = bootstrap_ci(values, n_boot, stable_seed("group", domain, method))
        group_rows.append({
            "domain": domain,
            "candidate_method": method,
            "rows": int(len(group)),
            "snapshots": int(group[keys].drop_duplicates().shape[0]),
            "proxy_cost_mean": float(values.mean()) if values.size else np.nan,
            "proxy_cost_ci95_lo": lo,
            "proxy_cost_ci95_hi": hi,
        })
    group_summary = pd.DataFrame(group_rows)

    paired_rows = []
    for domain in sorted(df["domain"].dropna().unique()):
        sub = df[df["domain"] == domain]
        pivot = sub.pivot_table(
            index=keys,
            columns="candidate_method",
            values="proxy_cost",
            aggfunc="mean",
        )
        if baseline_method not in pivot.columns:
            continue
        for method in sorted(col for col in pivot.columns if col != baseline_method):
            pair = pivot[[baseline_method, method]].dropna().reset_index()
            if pair.empty:
                continue
            pair["delta"] = pair[method].astype(float) - pair[baseline_method].astype(float)
            seed_delta = pair.groupby("seed")["delta"].mean().to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(seed_delta, n_boot, stable_seed("paired", domain, method, baseline_method))
            paired_rows.append({
                "domain": domain,
                "candidate_method": method,
                "baseline_method": baseline_method,
                "pairs": int(len(pair)),
                "seeds": int(pair["seed"].nunique()),
                f"delta_candidate_minus_{baseline_method}": float(seed_delta.mean()) if seed_delta.size else np.nan,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })
        best = pivot.min(axis=1).reset_index(name="oracle_best")
        base = pivot[[baseline_method]].reset_index()
        oracle = base.merge(best, on=keys, how="inner")
        oracle_delta_col = f"delta_oracle_minus_{baseline_method}"
        oracle[oracle_delta_col] = oracle["oracle_best"] - oracle[baseline_method]
        seed_delta = oracle.groupby("seed")[oracle_delta_col].mean().to_numpy(dtype=np.float64)
        lo, hi = bootstrap_ci(seed_delta, n_boot, stable_seed("oracle", domain, baseline_method))
        paired_rows.append({
            "domain": domain,
            "candidate_method": "oracle_best",
            "baseline_method": baseline_method,
            "pairs": int(len(oracle)),
            "seeds": int(oracle["seed"].nunique()),
            f"delta_candidate_minus_{baseline_method}": float(seed_delta.mean()) if seed_delta.size else np.nan,
            "ci95_lo": lo,
            "ci95_hi": hi,
        })
    paired_summary = pd.DataFrame(paired_rows)

    best = df.loc[df.groupby(keys)["proxy_cost"].idxmin(), keys + ["candidate_method", "proxy_cost"]]
    share = (
        best.groupby(["domain", "candidate_method"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    domain_counts = best.groupby("domain").size().rename("domain_snapshots").reset_index()
    share = share.merge(domain_counts, on="domain", how="left")
    share["win_share"] = share["wins"] / share["domain_snapshots"].clip(lower=1)
    return group_summary, paired_summary, share


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    df = read_label_files(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "snapshot_counterfactual_all.csv"
    df.to_csv(all_path, index=False)
    group_summary, paired_summary, win_share = summarize_labels(
        df,
        int(args.n_boot),
        str(args.baseline_method),
    )
    group_summary.to_csv(out_dir / "snapshot_counterfactual_group_summary.csv", index=False)
    paired_summary.to_csv(out_dir / "snapshot_counterfactual_paired_summary.csv", index=False)
    win_share.to_csv(out_dir / "snapshot_counterfactual_win_share.csv", index=False)
    print(f"WROTE {all_path}")
    print(paired_summary.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--burn-in-episodes", type=int, default=0)
    parser.add_argument("--upper-warmup-eps", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=20)
    parser.add_argument("--start-dispatch", type=int, default=4)
    parser.add_argument("--snapshot-stride", type=int, default=4)
    parser.add_argument("--horizon-s", type=float, default=600.0)
    parser.add_argument("--modes", default="target,terminalhold45")
    parser.add_argument("--deltas-s", default="-20,0,20")
    parser.add_argument("--terminal-hold-s", type=float, default=45.0)
    parser.add_argument("--terminal-min-s", type=float, default=0.0)
    parser.add_argument("--terminal-floor-ratio", type=float, default=0.0)
    parser.add_argument("--terminal-floor-min-s", type=float, default=0.0)
    parser.add_argument("--stochastic-lower", action="store_true")
    parser.add_argument("--out-dir", default="results_freqduet/snapshot_counterfactual_matrix")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--job-start", type=int, default=None)
    parser.add_argument("--job-end", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--baseline-method", default="target_0")
    args = parser.parse_args()

    configs = parse_csv(args.configs, str)
    seeds = parse_csv(args.seeds, int)
    if not args.aggregate_only:
        run_matrix(args, configs, seeds)
    if not args.no_aggregate:
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
