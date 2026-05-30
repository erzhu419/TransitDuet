#!/usr/bin/env python3
"""Evaluate FreqDuet demand-frequency modules on controlled synthetic streams.

This is not a policy-training benchmark. It tests whether the causal
decomposers expose the intended signals before we add more HRL machinery:

* low-frequency estimate tracks a known smooth demand trend
* high-frequency residual responds to injected burst shocks
* feature-allocation modes produce the expected policy-state dimensions
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frequency import DemandFrequencyTracker, fit_harmonic_prior


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_synthetic_stream(minutes=14 * 60, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(minutes, dtype=np.float64)

    # Smooth service-day structure: morning peak, midday plateau, evening peak.
    low = (
        10.0
        + 15.0 * _gaussian(t, 120.0, 55.0)
        + 8.0 * _gaussian(t, 420.0, 180.0)
        + 18.0 * _gaussian(t, 700.0, 70.0)
        + 2.5 * np.sin(2 * np.pi * t / minutes)
    )

    high = rng.normal(0.0, 1.0, size=minutes)
    burst_flag = np.zeros(minutes, dtype=bool)
    for center, amp, width in [
        (95, 13.0, 5),
        (260, -7.0, 4),
        (445, 11.0, 7),
        (610, 15.0, 5),
        (735, -8.0, 4),
    ]:
        pulse = amp * _gaussian(t, center, width)
        high += pulse
        burst_flag |= np.abs(pulse) > max(3.0, abs(amp) * 0.35)

    rate = np.clip(low + high, 0.1, None)
    # Observed counts per minute, split over two directions to exercise station
    # and OD paths without changing the known global target.
    observed = rng.poisson(rate)
    station_stream = []
    od_stream = []
    for y in observed:
        y0 = int(rng.binomial(int(y), 0.55)) if y > 0 else 0
        y1 = int(y) - y0
        station_stream.append({(1, True): y0, (2, False): y1})
        od_stream.append({
            (1, 5, True): y0,
            (2, 8, False): y1,
        })
    return low, high, burst_flag, station_stream, od_stream


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _f1(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def eval_method(method, low, high, burst_flag, station_stream, od_stream, args):
    tracker_method = "harmonic" if method == "harmonic_prior" else method
    harmonic_prior = None
    harmonic_prior_var = 100.0
    if method == "harmonic_prior":
        harmonic_prior = {
            "global": fit_harmonic_prior(
                low,
                update_interval_s=60.0,
                period_s=args.minutes * 60.0,
                fourier_k=args.fourier_k,
                ridge=args.harmonic_ridge,
            )
        }
        harmonic_prior_var = args.harmonic_prior_var
    tr = DemandFrequencyTracker(
        method=tracker_method,
        update_interval_s=60.0,
        bin_sec=args.bin_sec if tracker_method == "harmonic" else None,
        low_period_s=args.low_period_min * 60.0,
        fast_period_s=args.fast_period_min * 60.0,
        energy_period_s=args.energy_period_min * 60.0,
        forecast_horizon_s=args.forecast_min * 60.0,
        global_demand_norm=1.0,
        local_demand_norm=1.0,
        slope_norm=1.0,
        od_features=True,
        upper_mode="low",
        lower_mode="high",
        fourier_k=args.fourier_k,
        harmonic_forgetting=args.harmonic_forgetting,
        harmonic_prior_var=harmonic_prior_var,
        harmonic_prior=harmonic_prior,
    )
    pred_low = []
    pred_high = []
    pred_energy = []
    for st_counts, od_counts in zip(station_stream, od_stream):
        tr.update(st_counts, od_counts)
        pred_low.append(float(tr.global_state.low))
        pred_high.append(float(tr.global_state.high))
        pred_energy.append(math.sqrt(max(float(tr.global_state.high_energy), 0.0)))

    sl = slice(args.burn_in_min, None)
    low_err = np.asarray(pred_low)[sl] - low[sl]
    high_pred = np.asarray(pred_high)[sl]
    high_true = high[sl]
    burst_true = burst_flag[sl]

    n_true = int(max(np.sum(burst_true), 1))
    score = np.abs(high_pred)
    if score.size:
        cutoff = np.partition(score, max(score.size - n_true, 0))[max(score.size - n_true, 0)]
        burst_pred = score >= cutoff
    else:
        burst_pred = np.zeros_like(burst_true)
    precision, recall, f1 = _f1(burst_pred, burst_true)

    return {
        "method": method,
        "low_rmse": float(np.sqrt(np.mean(low_err ** 2))),
        "low_mae": float(np.mean(np.abs(low_err))),
        "high_corr": _corr(high_pred, high_true),
        "burst_precision": precision,
        "burst_recall": recall,
        "burst_f1": f1,
        "updates": int(tr.total_updates),
        "final_low": float(pred_low[-1]),
        "final_hf_energy": float(pred_energy[-1]),
    }


def check_feature_modes():
    rows = []
    for upper_mode, lower_mode, expected_u, expected_l in [
        ("low", "high", 6, 4),
        ("all", "all", 8, 7),
        ("high", "low", 5, 4),
    ]:
        tr = DemandFrequencyTracker(
            method="harmonic",
            update_interval_s=60.0,
            bin_sec=60,
            od_features=True,
            upper_mode=upper_mode,
            lower_mode=lower_mode,
        )
        tr.update({(1, True): 10}, {(1, 5, True): 10})
        u = tr.upper_features()
        l = tr.lower_features(1, True)
        rows.append({
            "upper_mode": upper_mode,
            "lower_mode": lower_mode,
            "upper_dim": int(len(u)),
            "lower_dim": int(len(l)),
            "expected_upper_dim": expected_u,
            "expected_lower_dim": expected_l,
            "pass": bool(len(u) == expected_u and len(l) == expected_l),
        })
    return rows


def print_table(rows):
    print("=" * 92)
    print(f"{'method':14s} {'low_rmse':>15s} {'low_mae':>15s} "
          f"{'high_corr':>15s} {'burst_f1':>15s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:14s} "
              f"{r['low_rmse_mean']:6.3f}±{r['low_rmse_std']:<6.3f} "
              f"{r['low_mae_mean']:6.3f}±{r['low_mae_std']:<6.3f} "
              f"{r['high_corr_mean']:6.3f}±{r['high_corr_std']:<6.3f} "
              f"{r['burst_f1_mean']:6.3f}±{r['burst_f1_std']:<6.3f}")
    print("=" * 92)


def aggregate_rows(per_seed_rows):
    methods = sorted({r["method"] for r in per_seed_rows})
    agg = []
    for method in methods:
        rows = [r for r in per_seed_rows if r["method"] == method]
        item = {"method": method, "n_seeds": len(rows)}
        for key in ["low_rmse", "low_mae", "high_corr", "burst_f1"]:
            vals = np.asarray([r[key] for r in rows], dtype=np.float64)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std())
        item["updates"] = int(rows[0]["updates"]) if rows else 0
        agg.append(item)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--minutes", type=int, default=14 * 60)
    ap.add_argument("--burn-in-min", type=int, default=90)
    ap.add_argument("--bin-sec", type=float, default=60.0)
    ap.add_argument("--low-period-min", type=float, default=60.0)
    ap.add_argument("--fast-period-min", type=float, default=5.0)
    ap.add_argument("--energy-period-min", type=float, default=10.0)
    ap.add_argument("--forecast-min", type=float, default=30.0)
    ap.add_argument("--fourier-k", type=int, default=6)
    ap.add_argument("--harmonic-forgetting", type=float, default=0.9995)
    ap.add_argument("--harmonic-prior-var", type=float, default=0.01)
    ap.add_argument("--harmonic-ridge", type=float, default=1e-2)
    ap.add_argument("--out", default="results_freqduet/frequency_module_eval.json")
    args = ap.parse_args()

    methods = ["harmonic", "harmonic_prior", "haar", "ema", "raw_history"]
    per_seed_rows = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        low, high, burst_flag, station_stream, od_stream = make_synthetic_stream(
            minutes=args.minutes, seed=seed)
        for method in methods:
            row = eval_method(
                method, low, high, burst_flag, station_stream, od_stream, args)
            row["seed"] = seed
            per_seed_rows.append(row)
    rows = aggregate_rows(per_seed_rows)
    mode_rows = check_feature_modes()

    print_table(rows)
    print("Feature allocation modes:")
    for r in mode_rows:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {status} upper={r['upper_mode']:4s} lower={r['lower_mode']:4s} "
              f"dims={r['upper_dim']}/{r['lower_dim']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "args": vars(args),
            "decomposer_metrics": rows,
            "per_seed_decomposer_metrics": per_seed_rows,
            "feature_mode_checks": mode_rows,
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
