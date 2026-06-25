#!/usr/bin/env python3
"""Summarize broad FreqDuet action-palette screens by config/action label.

`summarize_freqduet_paper_matrix.py` is intentionally paper-method oriented.
Palette screens need a looser view: compare many action configs directly,
without requiring every config name to map to a canonical paper method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("wait", "cv", "overshoot", "composite")


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


def infer_action(config: str) -> str:
    text = Path(str(config)).stem
    if "_paper_main_" in text or "cfaction_domainbest_v1" in text:
        return "paper_main_v1"
    if "cfactiontree" in text:
        return "cfactiontree"
    if "cfactionrule_v1" in text:
        return "cfactionrule_v1"
    if "cfaction_target_dm20_terminalonly" in text:
        return "target_m20_terminalonly"
    if "cfaction_target_dp0_terminalonly" in text:
        return "target_0_terminalonly"

    match = re.search(r"cfaction_v(?P<version>\d+)_(?P<mode>target|terminalhold\d+)_(?P<sign>d[mp])(?P<value>\d+)", text)
    if match:
        mode = match.group("mode")
        sign = "-" if match.group("sign") == "dm" else "+"
        value = int(match.group("value"))
        signed = 0 if value == 0 else f"{sign}{value}"
        return f"{mode}_{signed}"

    return text


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


def summarize_actions(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    for (domain, action), group in df.groupby(["domain", "action"], sort=False):
        if domain == "unknown":
            continue
        row = {
            "domain": domain,
            "action": action,
            "n_seeds": int(group["seed"].nunique()),
            "n_configs": int(group["config"].nunique()),
        }
        for metric in metrics:
            vals = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
            lo, hi = bootstrap_ci(
                vals, n_boot=n_boot,
                seed=stable_seed("summary", domain, action, metric),
            )
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)

    for action, group in df.groupby("action", sort=False):
        seed_action = (
            group.groupby("seed", as_index=False)[metrics]
            .mean(numeric_only=True)
        )
        if seed_action.empty:
            continue
        row = {
            "domain": "overall",
            "action": action,
            "n_seeds": int(seed_action["seed"].nunique()),
            "n_configs": int(group["config"].nunique()),
        }
        for metric in metrics:
            vals = seed_action[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
            lo, hi = bootstrap_ci(
                vals, n_boot=n_boot,
                seed=stable_seed("summary", "overall", action, metric),
            )
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(
    df: pd.DataFrame,
    baseline_action: str,
    metrics: list[str],
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    domains = [d for d in sorted(df["domain"].dropna().unique()) if d != "unknown"]
    domains.append("overall")
    actions = sorted(a for a in df["action"].dropna().unique() if a != baseline_action)
    for domain in domains:
        if domain == "overall":
            work = (
                df.groupby(["seed", "action"], as_index=False)[metrics]
                .mean(numeric_only=True)
            )
        else:
            work = df[df["domain"] == domain][["seed", "action", *metrics]].copy()
        for metric in metrics:
            pivot = work.pivot_table(index="seed", columns="action", values=metric, aggfunc="mean")
            if baseline_action not in pivot.columns:
                continue
            for action in actions:
                if action not in pivot.columns:
                    continue
                pair = pivot[[action, baseline_action]].dropna()
                if pair.empty:
                    continue
                delta = pair[action].astype(float) - pair[baseline_action].astype(float)
                lo, hi = bootstrap_ci(
                    delta.to_numpy(), n_boot=n_boot,
                    seed=stable_seed("delta", domain, action, baseline_action, metric),
                )
                rows.append({
                    "domain": domain,
                    "metric": metric,
                    "candidate_action": action,
                    "baseline_action": baseline_action,
                    "n_pairs": int(pair.shape[0]),
                    "candidate_mean": float(pair[action].mean()),
                    "baseline_mean": float(pair[baseline_action].mean()),
                    "delta_candidate_minus_baseline": float(delta.mean()),
                    "delta_ci95_lo": lo,
                    "delta_ci95_hi": hi,
                    "candidate_win_rate": float((delta < 0.0).mean()),
                })
    return pd.DataFrame(rows)


def best_rows(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for domain, group in summary.groupby("domain", sort=False):
        col = f"{metric}_mean"
        if col not in group.columns or group.empty:
            continue
        best = group.sort_values(col, ascending=True).iloc[0].to_dict()
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-action", default="paper_main_v1")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    df = pd.read_csv(args.per_seed)
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(f"missing metric columns: {missing}")
    df["domain"] = df["config"].astype(str).map(infer_domain)
    df["action"] = df["config"].astype(str).map(infer_action)
    df = df[df["domain"] != "unknown"].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_actions(df, metrics, n_boot=args.n_boot)
    deltas = paired_deltas(
        df, baseline_action=args.baseline_action,
        metrics=metrics, n_boot=args.n_boot,
    )
    best = best_rows(summary, "composite")
    summary.to_csv(out_dir / "palette_action_summary.csv", index=False)
    deltas.to_csv(out_dir / "palette_paired_deltas_vs_baseline.csv", index=False)
    best.to_csv(out_dir / "palette_best_by_domain.csv", index=False)
    payload = {
        "per_seed": args.per_seed,
        "baseline_action": args.baseline_action,
        "metrics": metrics,
        "n_rows": int(df.shape[0]),
        "n_configs": int(df["config"].nunique()),
        "n_actions": int(df["action"].nunique()),
    }
    with (out_dir / "palette_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_dir}")
    print(best[["domain", "action", "composite_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
