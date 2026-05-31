#!/usr/bin/env python3
"""Summarize FreqDuet demand trace logs.

Use this after running a config with ``frequency.logging.enable: true``. It
checks whether high-frequency demand energy aligns with lower holding and
passenger-wait spikes, which is the Phase-0 validation requested in
``md/dev_manual.md``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _corr(df, a, b):
    if a not in df.columns or b not in df.columns or len(df) < 3:
        return 0.0
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return 0.0
    if float(x[mask].std()) <= 1e-12 or float(y[mask].std()) <= 1e-12:
        return 0.0
    return float(x[mask].corr(y[mask]))


def _find_trace_files(paths, filename):
    result = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == filename:
            result.append(path)
        elif path.is_dir():
            direct = path / filename
            if direct.exists():
                result.append(direct)
            result.extend(path.glob(f"*/{filename}"))
    return sorted(set(result))


def summarize_run(trace_path, station_path=None):
    df = pd.read_csv(trace_path)
    row = {
        "run_dir": str(trace_path.parent),
        "rows": int(len(df)),
        "station_rows": 0,
        "mean_arrivals": float(df.get("arrivals", pd.Series(dtype=float)).mean()),
        "mean_queue_total": float(df.get("queue_total", pd.Series(dtype=float)).mean()),
        "mean_board_wait_s": float(
            df.get("board_wait_mean_s", pd.Series(dtype=float)).mean()),
        "mean_lower_action_s": float(
            df.get("lower_action_mean_s", pd.Series(dtype=float)).mean()),
        "mean_hf_energy": float(
            df.get("freq_high_energy", pd.Series(dtype=float)).mean()),
        "corr_hf_energy_action": _corr(
            df, "freq_high_energy", "lower_action_mean_s"),
        "corr_hf_energy_board_wait": _corr(
            df, "freq_high_energy", "board_wait_mean_s"),
        "corr_low_queue": _corr(df, "freq_low_demand", "queue_total"),
        "corr_middle_queue": _corr(df, "freq_middle", "queue_total"),
    }

    if station_path is not None and station_path.exists():
        sdf = pd.read_csv(station_path)
        active = sdf[
            (pd.to_numeric(sdf.get("arrivals", 0), errors="coerce") > 0)
            | (pd.to_numeric(sdf.get("boarded", 0), errors="coerce") > 0)
            | (pd.to_numeric(sdf.get("lower_action_count", 0), errors="coerce") > 0)
        ].copy()
        row.update({
            "station_rows": int(len(sdf)),
            "active_station_rows": int(len(active)),
            "corr_station_high_action": _corr(
                active, "local_high_energy", "lower_action_mean_s"),
            "corr_station_high_board_wait": _corr(
                active, "local_high_energy", "board_wait_mean_s"),
            "corr_station_high_queue": _corr(active, "local_high_energy", "queue"),
            "corr_station_low_queue": _corr(active, "local_low", "queue"),
        })
    else:
        row.update({
            "active_station_rows": 0,
            "corr_station_high_action": 0.0,
            "corr_station_high_board_wait": 0.0,
            "corr_station_high_queue": 0.0,
            "corr_station_low_queue": 0.0,
        })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run dirs, logs root, or demand_trace.csv files")
    parser.add_argument("--out", default=None, help="Optional CSV summary path")
    args = parser.parse_args()

    traces = _find_trace_files(args.paths, "demand_trace.csv")
    if not traces:
        raise SystemExit("No demand_trace.csv files found")
    rows = []
    for trace in traces:
        station = trace.parent / "demand_station_trace.csv"
        rows.append(summarize_run(trace, station))
    out = pd.DataFrame(rows)
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].round(6)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
