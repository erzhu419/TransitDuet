#!/usr/bin/env python3
"""Aggregate revision-2 final tables, including rule-baseline delay metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
EVAL_EP_MARKER = 9000

LEARNED = [
    "H_timetable_v5_stable_300",
    "H_hiro",
    "H_haar",
    "H_fixed_timetable_300",
    "H_fixed_timetable",
    "H_timetable_v4",
    "H_timetable_continuous",
    "H_timetable_no_context",
    "H_timetable_v4_independent_assembly",
]

RULES = [
    ("baseline_rule_daganzo", "rule_daganzo"),
    ("baseline_rule_xuan", "rule_xuan"),
]

MAIN_ROWS = [
    ("TransitDuet timetable", "H_timetable_v5_stable_300",
     "final rolling-timetable main method"),
    ("Target-headway only", "H_hiro",
     "legacy target-headway-only ablation"),
    ("HAAR-style RL proxy", "H_haar",
     "same-simulator RL coupling proxy"),
    ("Best fixed timetable SAC (holding-only, 300 s)", "H_fixed_timetable_300",
     "best fixed-headway holding-only SAC baseline from grid"),
    ("Fixed timetable SAC (holding-only, 360 s)", "H_fixed_timetable",
     "fixed 360-second timetable with same lower SAC"),
    ("Daganzo-style holding proxy", "baseline_rule_daganzo",
     "same-simulator analytical holding proxy; no training"),
    ("Xuan-style holding proxy", "baseline_rule_xuan",
     "same-simulator two-headway holding proxy; no training"),
]

EXTRA_COLS = [
    "avg_holding_sec",
    "avg_onboard_time_min",
    "avg_total_passenger_time_min",
    "planned_dispatch_headway_mean",
    "planned_dispatch_headway_std",
    "planned_dispatch_headway_cv",
    "actual_dispatch_headway_mean",
    "actual_dispatch_headway_std",
    "actual_dispatch_headway_cv",
    "planned_shift_mean",
    "planned_shift_std",
    "dispatch_lateness_mean",
    "dispatch_lateness_max",
    "upper_delta_std",
    "upper_delta_mean",
]


def composite_from_frame(df: pd.DataFrame) -> pd.Series:
    wait = pd.to_numeric(df["avg_wait_min"], errors="coerce")
    cv = pd.to_numeric(df["headway_cv"], errors="coerce")
    overshoot = pd.to_numeric(df.get("fleet_overshoot", 0.0), errors="coerce")
    n_fleet = pd.to_numeric(df.get("N_fleet", 12.0), errors="coerce").clip(lower=1.0)
    return wait / 10.0 + (overshoot ** 2) / n_fleet + cv


def summarize_numeric(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def read_learned(log_roots: list[Path], method: str, last_k: int) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        run_dir = None
        for root in log_roots:
            candidate = root / f"{method}_seed{seed}"
            if (candidate / "diagnostics.csv").exists():
                run_dir = candidate
                break
        if run_dir is None:
            continue
        df = pd.read_csv(run_dir / "diagnostics.csv")
        if "ep" in df.columns:
            df = df[df["ep"] < EVAL_EP_MARKER]
        if len(df) < last_k:
            continue
        tail = df.iloc[-last_k:].copy()
        comp = composite_from_frame(tail)
        row = {
            "method": method,
            "seed": seed,
            "n_ep": len(df),
            "last_k": last_k,
            "wait": float(pd.to_numeric(tail["avg_wait_min"]).mean()),
            "cv": float(pd.to_numeric(tail["headway_cv"]).mean()),
            "overshoot": float(pd.to_numeric(tail.get("fleet_overshoot", 0.0)).mean()),
            "fleet_pen": float((pd.to_numeric(tail.get("fleet_overshoot", 0.0)) ** 2
                                / pd.to_numeric(tail.get("N_fleet", 12.0)).clip(lower=1.0)).mean()),
            "composite": float(comp.mean()),
        }
        for col in EXTRA_COLS:
            if col in tail.columns:
                row[col] = float(pd.to_numeric(tail[col], errors="coerce").mean())
        rows.append(row)
    return rows


def read_rule(log_roots: list[Path], method: str, variant: str, last_k: int) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        run_dir = None
        for root in log_roots:
            candidate = root / f"baseline_{variant}_seed{seed}"
            if (candidate / "history.json").exists():
                run_dir = candidate
                break
        if run_dir is None:
            continue
        hist = json.loads((run_dir / "history.json").read_text())
        n_ep = len(hist.get("avg_wait", []))
        if n_ep < last_k:
            continue
        idx = slice(n_ep - last_k, n_ep)

        def mean_key(key: str, default: float = 0.0) -> float:
            vals = hist.get(key)
            if vals is None:
                return default
            return float(np.asarray(vals[idx], dtype=float).mean())

        row = {
            "method": method,
            "seed": seed,
            "n_ep": n_ep,
            "last_k": last_k,
            "wait": mean_key("avg_wait"),
            "cv": mean_key("cv"),
            "overshoot": mean_key("overshoot"),
            "fleet_pen": np.nan,
            "composite": mean_key("composite"),
        }
        for col in EXTRA_COLS:
            if col in hist:
                row[col] = mean_key(col)
        rows.append(row)
    return rows


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in per_seed.groupby("method", sort=False):
        row = {"method": method, "n_seed": int(g["seed"].nunique())}
        for col in [c for c in g.columns if c not in {"method", "seed", "n_ep", "last_k"}]:
            mean, std = summarize_numeric(g[col].tolist())
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-k", type=int, default=30)
    ap.add_argument("--logs", action="append", default=None)
    ap.add_argument("--out-dir", default="results_remote/revision2_final")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_roots = [Path(p) for p in (args.logs or ["logs", "logs_remote"])]
    log_roots = [p if p.is_absolute() else root / p for p in log_roots]

    rows: list[dict] = []
    for method in LEARNED:
        rows.extend(read_learned(log_roots, method, args.last_k))
    for method, variant in RULES:
        rows.extend(read_rule(log_roots, method, variant, args.last_k))

    if not rows:
        raise SystemExit("No matching runs found.")

    per_seed = pd.DataFrame(rows)
    aggregate_df = aggregate(per_seed)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.round(4).to_csv(out_dir / "per_seed_last30.csv", index=False)
    aggregate_df.round(4).to_csv(out_dir / "aggregate_last30.csv", index=False)

    main_rows = []
    for label, source, notes in MAIN_ROWS:
        found = aggregate_df[aggregate_df["method"] == source]
        if found.empty:
            continue
        r = found.iloc[0]
        main_rows.append({
            "method": label,
            "source": source,
            "n_seed": int(r["n_seed"]),
            "wait_mean": r.get("wait_mean", np.nan),
            "cv_mean": r.get("cv_mean", np.nan),
            "overshoot_mean": r.get("overshoot_mean", np.nan),
            "composite_mean": r.get("composite_mean", np.nan),
            "avg_holding_sec_mean": r.get("avg_holding_sec_mean", np.nan),
            "avg_onboard_time_min_mean": r.get("avg_onboard_time_min_mean", np.nan),
            "avg_total_passenger_time_min_mean": r.get("avg_total_passenger_time_min_mean", np.nan),
            "notes": notes,
        })
    main_df = pd.DataFrame(main_rows)
    main_df.round(4).to_csv(out_dir / "main_results_last30.csv", index=False)
    print(main_df.round(4).to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
