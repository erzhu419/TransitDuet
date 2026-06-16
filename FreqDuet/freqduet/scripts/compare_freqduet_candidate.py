#!/usr/bin/env python3
"""Compare a candidate FreqDuet method against the current paper matrix.

The usual paper longtrain matrix contains the promoted ``main`` plus internal
ablations. Repair candidates, such as drift-cost feedback, are often run as a
smaller matrix with only one method across the same domain/seed grid. This
script joins the two seed-level CSVs and reports paired deltas with bootstrap
confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
BASELINE_METHODS = ("main", "nofreq", "rawhistory", "allfreq", "nopromotion", "noleakage")
CANDIDATE_METHOD_TOKENS = (
    "snapshottermbias_m01",
    "snapshottermbias",
    "snapshotrf_m02",
    "snapshotrf",
    "cfvalue_multicand",
    "cfvalue_noisegate",
    "cfvalue_domainmean",
    "upperres_planctx",
    "upperres_reliefguard",
    "upperres_selector",
    "uppervalue_hfgate",
    "termvalselector",
    "headwayplanner",
    "releaseadapt",
    "release20",
    "release15",
    "release10",
    "release5",
    "ctxselector60",
    "upperdisc7",
    "upperdisc5",
    "upperdisc4",
    "upperdisc3",
    "spline2dir_promreplan",
    "spacectx_disc9",
    "gapctx_disc9last",
    "disc9last",
    "spacectx",
    "disc9",
    "sumorl_rawhist_holdrl",
    "sumorl_holdrl",
    "histaux6eg06upper",
    "histaux6eg06",
    "histaux6eg05",
    "histaux6",
    "histaux3",
    "termvalue20",
    "fixedselector_balanced",
    "fixedselector",
    "headfloor100",
    "headfloor095",
    "valuesoft",
    "termrelief20",
    "termfb30",
    "termhold45",
    "valueguard",
)
DEFAULT_METRICS = (
    "wait",
    "cv",
    "overshoot",
    "composite",
    "lower_action_mean",
    "lower_drift_penalty_mean",
    "lower_drift_cost_mean",
    "lower_drift_cost_adaptive_gate_mean",
    "upper_residual_value_cost_mean",
    "upper_residual_value_cost_max",
    "upper_residual_value_cost_active_mean",
    "upper_residual_selector_active_mean",
    "upper_residual_selector_adjust_mean",
    "upper_residual_selector_adjust_max",
    "upper_residual_selector_margin_mean",
    "upper_residual_selector_updates",
    "headway_value_planner_active_mean",
    "headway_value_planner_adjust_mean",
    "headway_value_planner_adjust_max",
    "headway_value_planner_delta_mean",
    "headway_value_planner_delta_max",
    "headway_value_planner_margin_mean",
    "headway_value_planner_actor_pred_mean",
    "headway_value_planner_selected_pred_mean",
    "headway_value_planner_prior_mean",
    "headway_value_planner_target_cost_mean",
    "headway_value_planner_target_cost_max",
    "headway_value_planner_updates",
    "w_fleet",
    "theta_fleet",
    "upper_delta_mean",
    "upper_plan_target_mean",
    "upper_plan_reuse_ratio",
    "terminal_launch_shift_mean",
    "terminal_launch_shift_std",
    "terminal_shift_cap_mean",
    "terminal_shift_min_mean",
    "terminal_shift_min_min",
    "terminal_feedback_bias_mean",
    "terminal_feedback_bias_max",
    "terminal_feedback_events",
    "terminal_value_selector_active_mean",
    "terminal_value_selector_bias_mean",
    "terminal_value_selector_bias_max",
    "terminal_value_selector_margin_mean",
    "terminal_value_selector_target_cost_mean",
    "terminal_value_selector_target_cost_max",
    "terminal_value_selector_updates",
    "terminal_headway_floor_mean",
    "terminal_headway_floor_events",
    "fixed_selector_fixed_active",
    "fixed_selector_learned_cost_ema",
    "fixed_selector_fixed_cost_ema",
    "fixed_selector_learned_count",
    "fixed_selector_fixed_count",
    "fixed_selector_context_enabled",
    "fixed_selector_context_learned_value",
    "fixed_selector_context_fixed_value",
    "fixed_selector_context_margin",
    "fixed_selector_context_feature_norm",
    "snapshot_value_selector_enabled",
    "snapshot_value_active_mean",
    "snapshot_value_events",
    "snapshot_value_changed_mean",
    "snapshot_value_changed_events",
    "snapshot_value_override_mean",
    "snapshot_value_override_events",
    "snapshot_value_terminal_dispatch_mean",
    "snapshot_value_terminal_dispatch_events",
    "snapshot_value_terminal_bias_mean",
    "snapshot_value_terminal_bias_max",
    "snapshot_value_terminal_bias_events",
    "snapshot_value_margin_mean",
    "snapshot_value_margin_max",
    "snapshot_value_pred_mean",
    "snapshot_value_baseline_pred_mean",
    "fleet_noharm_upper_adjust_mean",
    "fleet_noharm_upper_gate_active_mean",
    "fleet_noharm_lower_adjust_mean",
    "fleet_noharm_lower_gate_active_mean",
    "fleet_noharm_lower_proactive_adjust_mean",
    "fleet_noharm_lower_proactive_gate_active_mean",
    "fleet_noharm_lower_value_guard_adjust_mean",
    "fleet_noharm_lower_value_guard_active_mean",
    "fleet_noharm_lower_value_guard_value_mean",
    "fleet_noharm_lower_value_guard_headway_mean",
    "fleet_noharm_lower_value_guard_cost_mean",
    "fleet_noharm_lower_value_soft_cost_mean",
    "fleet_noharm_lower_value_soft_cost_max",
    "fleet_noharm_lower_value_soft_events",
    "fleet_noharm_lower_value_soft_active_mean",
    "fleet_noharm_lower_value_soft_value_mean",
    "fleet_noharm_lower_value_soft_headway_mean",
    "fleet_noharm_lower_value_soft_risk_mean",
    "fleet_noharm_lower_value_soft_violation_mean",
)


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


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


def infer_method(config: str) -> str:
    for token in CANDIDATE_METHOD_TOKENS:
        if token in config:
            return token
    if "driftcost" in config:
        return "main_driftcost"
    if config.endswith("_main_hiro"):
        return "main"
    for method in BASELINE_METHODS:
        if method != "main" and f"_{method}_" in config:
            return method
    return "unknown"


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"config", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{source} missing required columns: {missing}")
    out = df.copy()
    out["domain"] = out["config"].astype(str).map(infer_domain)
    out["method"] = out["config"].astype(str).map(infer_method)
    out["source"] = source
    return out


def domain_seed_metric(df: pd.DataFrame, metric: str, domain: str) -> pd.DataFrame:
    if domain == "overall":
        return df.groupby(["seed", "method"], as_index=False)[metric].mean()
    return df.loc[df["domain"] == domain, ["seed", "method", metric]].copy()


def summarize_methods(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    for (domain, method), group in df.groupby(["domain", "method"], sort=False):
        if domain == "unknown" or method == "unknown":
            continue
        row = {"domain": domain, "method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=0)) if vals.size else np.nan
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=stable_seed("summary", domain, method, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)

    for method in sorted(df["method"].dropna().unique()):
        if method == "unknown":
            continue
        sub = df[df["method"] == method]
        row = {"domain": "overall", "method": method, "n_seeds": int(sub["seed"].nunique())}
        for metric in metrics:
            seed_metric = (
                sub.groupby(["seed", "domain"], as_index=False)[metric]
                .mean()
                .groupby("seed", as_index=False)[metric]
                .mean()
            )
            vals = pd.to_numeric(seed_metric[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=0)) if vals.size else np.nan
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=stable_seed("summary", "overall", method, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(
    df: pd.DataFrame,
    candidate_method: str,
    baselines: list[str],
    metrics: list[str],
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for domain in DOMAINS:
            metric_df = domain_seed_metric(df, metric, domain)
            pivot = metric_df.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
            if candidate_method not in pivot.columns:
                continue
            for baseline in baselines:
                if baseline == candidate_method or baseline not in pivot.columns:
                    continue
                pair = pivot[[candidate_method, baseline]].dropna()
                if pair.empty:
                    continue
                delta = pair[candidate_method].astype(float) - pair[baseline].astype(float)
                lo, hi = bootstrap_ci(
                    delta.to_numpy(),
                    n_boot=n_boot,
                    seed=stable_seed("delta", domain, candidate_method, baseline, metric),
                )
                rows.append({
                    "domain": domain,
                    "metric": metric,
                    "candidate": candidate_method,
                    "baseline": baseline,
                    "n_pairs": int(len(pair)),
                    "candidate_mean": float(pair[candidate_method].mean()),
                    "baseline_mean": float(pair[baseline].mean()),
                    "delta_candidate_minus_baseline": float(delta.mean()),
                    "delta_ci95_lo": lo,
                    "delta_ci95_hi": hi,
                    "candidate_win_rate": float((delta < 0.0).mean()),
                    "candidate_tie_rate": float((delta == 0.0).mean()),
                })
        for baseline in baselines:
            if baseline == candidate_method:
                continue
            cand = df[
                (df["method"] == candidate_method)
                & (df["domain"].isin(DOMAINS))
            ][["seed", "domain", metric]].copy()
            base = df[
                (df["method"] == baseline)
                & (df["domain"].isin(DOMAINS))
            ][["seed", "domain", metric]].copy()
            pair = cand.merge(
                base,
                on=["seed", "domain"],
                suffixes=("_candidate", "_baseline"),
            ).dropna()
            if pair.empty:
                continue
            pair["delta"] = (
                pair[f"{metric}_candidate"].astype(float)
                - pair[f"{metric}_baseline"].astype(float)
            )
            seed_pair = pair.groupby("seed", as_index=False).agg(
                candidate_mean=(f"{metric}_candidate", "mean"),
                baseline_mean=(f"{metric}_baseline", "mean"),
                delta=("delta", "mean"),
            )
            delta = seed_pair["delta"].astype(float)
            lo, hi = bootstrap_ci(
                delta.to_numpy(),
                n_boot=n_boot,
                seed=stable_seed("delta", "overall", candidate_method, baseline, metric),
            )
            rows.append({
                "domain": "overall",
                "metric": metric,
                "candidate": candidate_method,
                "baseline": baseline,
                "n_pairs": int(len(seed_pair)),
                "candidate_mean": float(seed_pair["candidate_mean"].mean()),
                "baseline_mean": float(seed_pair["baseline_mean"].mean()),
                "delta_candidate_minus_baseline": float(delta.mean()),
                "delta_ci95_lo": lo,
                "delta_ci95_hi": hi,
                "candidate_win_rate": float((delta < 0.0).mean()),
                "candidate_tie_rate": float((delta == 0.0).mean()),
            })
    return pd.DataFrame(rows)


def print_compact(deltas: pd.DataFrame, metric: str) -> None:
    rows = deltas[deltas["metric"] == metric].copy()
    if rows.empty:
        print("No paired deltas available.")
        return
    print("=" * 116)
    print(f"{'domain':12s} {'baseline':12s} {'n':>4s} {'cand':>9s} {'base':>9s} {'delta':>10s} {'ci95':>23s} {'win':>7s}")
    print("-" * 116)
    order = {d: i for i, d in enumerate([*DOMAINS, "overall"])}
    rows["_rank"] = rows["domain"].map(order).fillna(99)
    rows = rows.sort_values(["_rank", "baseline"])
    for _, row in rows.iterrows():
        print(
            f"{row['domain']:12s} {row['baseline']:12s} {int(row['n_pairs']):4d} "
            f"{row['candidate_mean']:9.4f} {row['baseline_mean']:9.4f} "
            f"{row['delta_candidate_minus_baseline']:10.4f} "
            f"[{row['delta_ci95_lo']:+.4f},{row['delta_ci95_hi']:+.4f}] "
            f"{row['candidate_win_rate']:7.3f}"
        )
    print("=" * 116)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-per-seed", required=True, help="Current paper matrix freqduet_ablation_per_seed.csv")
    ap.add_argument("--candidate-per-seed", required=True, help="Candidate matrix freqduet_ablation_per_seed.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidate-method", default="main_driftcost")
    ap.add_argument("--baselines", default=",".join(BASELINE_METHODS))
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--paired-metric", default="composite")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    baseline = prepare(pd.read_csv(args.baseline_per_seed), "baseline")
    candidate = prepare(pd.read_csv(args.candidate_per_seed), "candidate")
    candidate = candidate[candidate["domain"] != "unknown"].copy()
    matched = candidate["method"] == args.candidate_method
    if matched.any():
        candidate = candidate[matched].copy()
    else:
        known_methods = sorted(
            m for m in candidate["method"].dropna().unique()
            if m != "unknown"
        )
        if known_methods:
            raise SystemExit(
                "candidate file contains inferred methods but none match "
                f"{args.candidate_method!r}: {known_methods}"
            )
        candidate["method"] = args.candidate_method

    duplicates = candidate.duplicated(["domain", "seed"], keep=False)
    if duplicates.any():
        bad = (
            candidate.loc[duplicates, ["domain", "seed", "config"]]
            .sort_values(["domain", "seed", "config"])
            .head(20)
        )
        raise SystemExit(
            "candidate has duplicate domain/seed rows after method filtering; "
            f"first duplicates:\n{bad.to_string(index=False)}"
        )

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]
    missing_metrics = [m for m in metrics if m not in baseline.columns or m not in candidate.columns]
    if args.paired_metric not in metrics:
        raise SystemExit(f"paired metric {args.paired_metric!r} not included in metrics {metrics}")
    if args.paired_metric not in baseline.columns or args.paired_metric not in candidate.columns:
        raise SystemExit(f"paired metric {args.paired_metric!r} missing from at least one input")
    if missing_metrics:
        print("Warning: skipping metrics missing from at least one input:")
        for metric in missing_metrics:
            print(f"  {metric}")
        metrics = [m for m in metrics if m not in missing_metrics]

    combined = pd.concat([baseline, candidate], ignore_index=True, sort=False)
    unknown = combined[(combined["domain"] == "unknown") | (combined["method"] == "unknown")]["config"].unique()
    if len(unknown):
        print("Warning: ignored unknown config names:")
        for name in sorted(unknown):
            print(f"  {name}")
    combined = combined[(combined["domain"] != "unknown") & (combined["method"] != "unknown")].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_summary = summarize_methods(combined, metrics, n_boot=args.n_boot)
    deltas = paired_deltas(combined, args.candidate_method, baselines, metrics, n_boot=args.n_boot)
    method_summary.to_csv(out_dir / "candidate_method_summary.csv", index=False)
    deltas.to_csv(out_dir / "candidate_paired_deltas.csv", index=False)

    payload = {
        "baseline_per_seed": str(args.baseline_per_seed),
        "candidate_per_seed": str(args.candidate_per_seed),
        "candidate_method": args.candidate_method,
        "baselines": baselines,
        "metrics": metrics,
        "paired_metric": args.paired_metric,
        "n_rows": int(len(combined)),
        "n_boot": int(args.n_boot),
    }
    with (out_dir / "candidate_comparison.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print_compact(deltas, args.paired_metric)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
