#!/usr/bin/env python3
"""Paired comparison between a FreqDuet candidate and external baselines."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
DEFAULT_METRICS = ("wait", "cv", "overshoot", "composite")
CANDIDATE_METHOD_TOKENS = (
    "freeze100",
    "snapshotzerobias",
    "snapshotrushriskgate_tight7",
    "snapshotrushriskgate_soft7",
    "snapshotrushriskgate_loose0",
    "snapshotrushriskgate",
    "snapshotactorrel_targetonly_m02",
    "snapshotactorrel_targetonly",
    "snapshotactorrel",
    "snapshotriskdual_targetnonpos",
    "snapshotriskdual_targetguard_b15o30",
    "snapshotriskdual_targetguard_b15o22",
    "snapshotriskdual_targetguard_b5o22",
    "snapshotriskdual_targetguard",
    "snapshotriskdual_targetonly",
    "snapshotriskdual",
    "snapshotriskvalue_targetonly",
    "snapshotriskvalue",
    "snapshotriskmix",
    "snapshotriskpenalty",
    "snapshotodtermcap",
    "snapshotdomaincap",
    "snapshotnoisegate",
    "snapshottermbias_cap15",
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
    "releaseguard",
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
    "spacectx_causalgate",
    "spacectx_disc9",
    "gapctx_disc9last",
    "disc9last",
    "spacectx",
    "disc9",
    "fixedselector_balanced",
    "sumorl_rawhist_holdrl",
    "sumorl_holdrl",
    "histaux6eg06upper",
    "histaux6eg06",
    "histaux6eg05",
    "histaux6",
    "histaux3",
    "termvalue20",
    "fixedselector",
    "headfloor100",
    "headfloor095",
    "valuesoft35_lfsafe",
    "valuesoft35",
    "valuesoft",
    "termrelief20",
    "termfb30",
    "termhold45",
    "valueguard",
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
    text = str(config)
    for token in CANDIDATE_METHOD_TOKENS:
        if token in text:
            return token
    if "driftcost" in text:
        return "main_driftcost"
    if text.endswith("_main_hiro"):
        return "main"
    for method in ("nofreq", "rawhistory", "allfreq", "nopromotion",
                   "noleakage"):
        if f"_{method}_" in text:
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


def prepare_candidate(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"config", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"candidate missing required columns: {missing}")
    if "domain" not in df.columns:
        df["domain"] = df["config"].astype(str).map(infer_domain)
    inferred = df["config"].astype(str).map(infer_method)
    matched = inferred == method
    if matched.any():
        df = df[matched].copy()
    elif inferred.nunique(dropna=True) > 1:
        counts = inferred.value_counts().to_dict()
        raise SystemExit(
            "candidate file contains multiple inferred methods but none "
            f"match {method!r}: {counts}"
        )
    duplicates = df.duplicated(["domain", "seed"], keep=False)
    if duplicates.any():
        bad = (
            df.loc[duplicates, ["domain", "seed", "config"]]
            .sort_values(["domain", "seed", "config"])
            .head(20)
        )
        raise SystemExit(
            "candidate has duplicate domain/seed rows after method filtering; "
            f"first duplicates:\n{bad.to_string(index=False)}"
        )
    df["method"] = method
    return df[df["domain"].isin(DOMAINS)].copy()


def prepare_baseline(path: Path, methods: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"domain", "method", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"baseline missing required columns: {missing}")
    return df[df["domain"].isin(DOMAINS) & df["method"].isin(methods)].copy()


def paired_deltas(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate_method: str,
    baseline_methods: list[str],
    metrics: list[str],
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for baseline_method in baseline_methods:
        base_method = baseline[baseline["method"] == baseline_method].copy()
        for domain in DOMAINS:
            cand_domain = candidate[candidate["domain"] == domain]
            base_domain = base_method[base_method["domain"] == domain]
            pair = cand_domain.merge(
                base_domain,
                on=["domain", "seed"],
                suffixes=("_candidate", "_baseline"),
            )
            if pair.empty:
                continue
            row = {
                "domain": domain,
                "candidate": candidate_method,
                "baseline": baseline_method,
                "n_pairs": int(len(pair)),
            }
            for metric in metrics:
                delta = (
                    pair[f"{metric}_candidate"].astype(float)
                    - pair[f"{metric}_baseline"].astype(float)
                )
                lo, hi = bootstrap_ci(
                    delta.to_numpy(),
                    n_boot=n_boot,
                    seed=stable_seed("external", domain, candidate_method, baseline_method, metric),
                )
                row[f"{metric}_candidate_mean"] = float(pair[f"{metric}_candidate"].mean())
                row[f"{metric}_baseline_mean"] = float(pair[f"{metric}_baseline"].mean())
                row[f"{metric}_delta_mean"] = float(delta.mean())
                row[f"{metric}_delta_ci95_lo"] = lo
                row[f"{metric}_delta_ci95_hi"] = hi
                row[f"{metric}_win_rate"] = float((delta < 0.0).mean())
            rows.append(row)

        pair = candidate.merge(
            base_method,
            on=["domain", "seed"],
            suffixes=("_candidate", "_baseline"),
        )
        if pair.empty:
            continue
        row = {
            "domain": "overall_shared",
            "candidate": candidate_method,
            "baseline": baseline_method,
            "n_pairs": int(len(pair)),
        }
        for metric in metrics:
            pair_metric = pair[["domain", "seed", f"{metric}_candidate", f"{metric}_baseline"]].copy()
            pair_metric["delta"] = (
                pair_metric[f"{metric}_candidate"].astype(float)
                - pair_metric[f"{metric}_baseline"].astype(float)
            )
            seed_delta = pair_metric.groupby("seed", as_index=False).agg(
                candidate_mean=(f"{metric}_candidate", "mean"),
                baseline_mean=(f"{metric}_baseline", "mean"),
                delta=("delta", "mean"),
            )
            delta = seed_delta["delta"].astype(float)
            lo, hi = bootstrap_ci(
                delta.to_numpy(),
                n_boot=n_boot,
                seed=stable_seed("external", "overall_shared", candidate_method, baseline_method, metric),
            )
            row[f"{metric}_candidate_mean"] = float(seed_delta["candidate_mean"].mean())
            row[f"{metric}_baseline_mean"] = float(seed_delta["baseline_mean"].mean())
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_delta_ci95_lo"] = lo
            row[f"{metric}_delta_ci95_hi"] = hi
            row[f"{metric}_win_rate"] = float((delta < 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def print_compact(rows: pd.DataFrame, metric: str) -> None:
    if rows.empty:
        print("No paired rows.")
        return
    print("=" * 116)
    print(f"{'domain':15s} {'baseline':14s} {'n':>4s} {'cand':>9s} {'base':>9s} {'delta':>10s} {'ci95':>23s} {'win':>7s}")
    print("-" * 116)
    order = {d: i for i, d in enumerate([*DOMAINS, "overall_shared"])}
    rows = rows.copy()
    rows["_rank"] = rows["domain"].map(order).fillna(99)
    rows = rows.sort_values(["_rank", "baseline"])
    for _, row in rows.iterrows():
        print(
            f"{row['domain']:15s} {row['baseline']:14s} {int(row['n_pairs']):4d} "
            f"{row[f'{metric}_candidate_mean']:9.4f} "
            f"{row[f'{metric}_baseline_mean']:9.4f} "
            f"{row[f'{metric}_delta_mean']:10.4f} "
            f"[{row[f'{metric}_delta_ci95_lo']:+.4f},{row[f'{metric}_delta_ci95_hi']:+.4f}] "
            f"{row[f'{metric}_win_rate']:7.3f}"
        )
    print("=" * 116)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-per-seed", required=True)
    ap.add_argument("--baseline-per-seed", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidate-method", required=True)
    ap.add_argument("--baseline-methods", default="fixed_headway")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--paired-metric", default="composite")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    baseline_methods = [m.strip() for m in args.baseline_methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.paired_metric not in metrics:
        raise SystemExit(f"paired metric {args.paired_metric!r} not included in metrics {metrics}")

    candidate = prepare_candidate(Path(args.candidate_per_seed), args.candidate_method)
    baseline = prepare_baseline(Path(args.baseline_per_seed), baseline_methods)
    missing = [
        metric for metric in metrics
        if metric not in candidate.columns or metric not in baseline.columns
    ]
    if args.paired_metric in missing:
        raise SystemExit(f"paired metric {args.paired_metric!r} missing from input")
    if missing:
        print("Warning: skipping metrics missing from at least one input:")
        for metric in missing:
            print(f"  {metric}")
        metrics = [m for m in metrics if m not in missing]

    rows = paired_deltas(
        candidate,
        baseline,
        args.candidate_method,
        baseline_methods,
        metrics,
        n_boot=args.n_boot,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "external_baseline_paired_deltas.csv", index=False)
    with (out_dir / "external_baseline_comparison.json").open("w") as f:
        json.dump({
            "candidate_per_seed": args.candidate_per_seed,
            "baseline_per_seed": args.baseline_per_seed,
            "candidate_method": args.candidate_method,
            "baseline_methods": baseline_methods,
            "metrics": metrics,
            "paired_metric": args.paired_metric,
            "n_boot": int(args.n_boot),
        }, f, indent=2)
    print_compact(rows, args.paired_metric)


if __name__ == "__main__":
    main()
