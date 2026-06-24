#!/usr/bin/env python3
"""Summarize fixed-action counterfactual rollouts for FreqDuet.

The action counterfactual matrix holds the learned policy fixed except for a
config-gated upper action override. This script parses those config names and
reports seed-paired deltas, so a fixed delta or terminal hold candidate can be
judged under common random numbers instead of raw means.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
MODES = ("target", "terminalhold45")
DEFAULT_METRICS = (
    "wait",
    "cv",
    "overshoot",
    "composite",
    "upper_delta_mean",
    "upper_delta_std",
    "terminal_launch_shift_mean",
    "terminal_launch_shift_std",
    "shock_response_time_mean_s",
    "lower_action_mean",
    "freq_promotion_ratio",
)
CONFIG_RE = re.compile(
    r"^F_freqduet_(?P<domain>terminal|gen_highnoise|gen_odshift|gen_rushshift)"
    r"_main_cfaction_v\d+_(?P<mode>target|terminalhold45)_(?P<delta>d[mp]\d+)_hiro$"
)


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


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


def parse_domain(token: str) -> str:
    return {
        "terminal": "terminal",
        "gen_highnoise": "highnoise",
        "gen_odshift": "odshift",
        "gen_rushshift": "rushshift",
    }.get(token, "unknown")


def parse_delta_s(token: str) -> int:
    sign = -1 if token.startswith("dm") else 1
    return sign * int(token[2:])


def parse_config(config: str) -> dict[str, object]:
    match = CONFIG_RE.match(str(config))
    if not match:
        return {"domain": "unknown", "mode": "unknown", "delta_tag": "unknown", "delta_s": np.nan}
    delta_tag = match.group("delta")
    return {
        "domain": parse_domain(match.group("domain")),
        "mode": match.group("mode"),
        "delta_tag": delta_tag,
        "delta_s": parse_delta_s(delta_tag),
    }


def metric_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    metrics = requested or list(DEFAULT_METRICS)
    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise SystemExit(f"missing metric columns: {missing}")
    return metrics


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {"config", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"input missing required columns: {missing}")
    parsed = pd.DataFrame([parse_config(config) for config in df["config"]])
    out = pd.concat([df.reset_index(drop=True), parsed], axis=1)
    out = out[(out["domain"] != "unknown") & (out["mode"] != "unknown")].copy()
    out["seed"] = out["seed"].astype(int)
    out["delta_s"] = out["delta_s"].astype(int)
    return out


def with_overall(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keep = ["domain", "mode", "delta_s", "delta_tag", "seed", *metrics]
    base = df[keep].copy()
    overall = (
        df.groupby(["seed", "mode", "delta_s", "delta_tag"], as_index=False)[metrics]
        .mean()
        .assign(domain="overall")
    )
    return pd.concat([base, overall[keep]], ignore_index=True)


def summarize_groups(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, mode, delta_s), group in df.groupby(["domain", "mode", "delta_s"], sort=True):
        row: dict[str, object] = {
            "domain": domain,
            "mode": mode,
            "delta_s": int(delta_s),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(vals.mean()) if vals.size else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=0)) if vals.size else np.nan
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=stable_seed("group", domain, mode, delta_s, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_zero(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for domain in (*DOMAINS, "overall"):
            for mode in MODES:
                sub = df[(df["domain"] == domain) & (df["mode"] == mode)]
                pivot = sub.pivot_table(index="seed", columns="delta_s", values=metric, aggfunc="mean")
                if 0 not in pivot.columns:
                    continue
                for delta_s in sorted(col for col in pivot.columns if col != 0):
                    pair = pivot[[0, delta_s]].dropna()
                    if pair.empty:
                        continue
                    delta = pair[delta_s].astype(float) - pair[0].astype(float)
                    lo, hi = bootstrap_ci(
                        delta.to_numpy(),
                        n_boot=n_boot,
                        seed=stable_seed("vs_zero", domain, mode, delta_s, metric),
                    )
                    rows.append({
                        "domain": domain,
                        "mode": mode,
                        "metric": metric,
                        "candidate_delta_s": int(delta_s),
                        "baseline_delta_s": 0,
                        "n_pairs": int(len(pair)),
                        "candidate_mean": float(pair[delta_s].mean()),
                        "baseline_mean": float(pair[0].mean()),
                        "delta_candidate_minus_zero": float(delta.mean()),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    })
    return pd.DataFrame(rows)


def paired_modes(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for domain in (*DOMAINS, "overall"):
            for delta_s in sorted(df["delta_s"].dropna().unique()):
                sub = df[(df["domain"] == domain) & (df["delta_s"] == delta_s)]
                pivot = sub.pivot_table(index="seed", columns="mode", values=metric, aggfunc="mean")
                if "target" not in pivot.columns or "terminalhold45" not in pivot.columns:
                    continue
                pair = pivot[["target", "terminalhold45"]].dropna()
                if pair.empty:
                    continue
                delta = pair["terminalhold45"].astype(float) - pair["target"].astype(float)
                lo, hi = bootstrap_ci(
                    delta.to_numpy(),
                    n_boot=n_boot,
                    seed=stable_seed("mode", domain, delta_s, metric),
                )
                rows.append({
                    "domain": domain,
                    "delta_s": int(delta_s),
                    "metric": metric,
                    "candidate_mode": "terminalhold45",
                    "baseline_mode": "target",
                    "n_pairs": int(len(pair)),
                    "candidate_mean": float(pair["terminalhold45"].mean()),
                    "baseline_mean": float(pair["target"].mean()),
                    "delta_terminalhold45_minus_target": float(delta.mean()),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                })
    return pd.DataFrame(rows)


def best_by_domain(df: pd.DataFrame, group_summary: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain in (*DOMAINS, "overall"):
        sub_summary = group_summary[group_summary["domain"] == domain].copy()
        if sub_summary.empty:
            continue
        best = sub_summary.loc[sub_summary["composite_mean"].idxmin()]
        base = df[
            (df["domain"] == domain)
            & (df["mode"] == "target")
            & (df["delta_s"] == 0)
        ][["seed", "composite"]].rename(columns={"composite": "target_zero"})
        cand = df[
            (df["domain"] == domain)
            & (df["mode"] == best["mode"])
            & (df["delta_s"] == best["delta_s"])
        ][["seed", "composite"]].rename(columns={"composite": "candidate"})
        pair = cand.merge(base, on="seed", how="inner")
        if pair.empty:
            delta_mean = np.nan
            lo = hi = np.nan
            n_pairs = 0
            target_zero_mean = np.nan
        else:
            delta = pair["candidate"].astype(float) - pair["target_zero"].astype(float)
            lo, hi = bootstrap_ci(
                delta.to_numpy(),
                n_boot=n_boot,
                seed=stable_seed("best", domain, str(best["mode"]), int(best["delta_s"])),
            )
            delta_mean = float(delta.mean())
            target_zero_mean = float(pair["target_zero"].mean())
            n_pairs = int(len(pair))
        rows.append({
            "domain": domain,
            "best_mode": str(best["mode"]),
            "best_delta_s": int(best["delta_s"]),
            "best_composite_mean": float(best["composite_mean"]),
            "target_zero_composite_mean": target_zero_mean,
            "delta_best_minus_target_zero": delta_mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_pairs": n_pairs,
        })
    return pd.DataFrame(rows)


def print_compact(group_summary: pd.DataFrame, vs_zero: pd.DataFrame, mode_deltas: pd.DataFrame, best: pd.DataFrame) -> None:
    cols = ["domain", "mode", "delta_s", "n_seeds", "composite_mean", "wait_mean", "cv_mean", "upper_delta_mean_mean", "terminal_launch_shift_mean_mean"]
    available = [col for col in cols if col in group_summary.columns]
    print("\n== composite group summary ==")
    print(group_summary[available].sort_values(["domain", "mode", "delta_s"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    key = vs_zero[vs_zero["metric"] == "composite"].copy()
    print("\n== paired composite delta vs same-mode delta0 (negative is better) ==")
    if key.empty:
        print("none")
    else:
        print(
            key.sort_values(["domain", "mode", "candidate_delta_s"])[
                ["domain", "mode", "candidate_delta_s", "n_pairs", "delta_candidate_minus_zero", "ci95_lo", "ci95_hi"]
            ].to_string(index=False, float_format=lambda x: f"{x:+.4f}")
        )

    key = mode_deltas[mode_deltas["metric"] == "composite"].copy()
    print("\n== paired composite terminalhold45 - target (negative is better) ==")
    if key.empty:
        print("none")
    else:
        print(
            key.sort_values(["domain", "delta_s"])[
                ["domain", "delta_s", "n_pairs", "delta_terminalhold45_minus_target", "ci95_lo", "ci95_hi"]
            ].to_string(index=False, float_format=lambda x: f"{x:+.4f}")
        )

    print("\n== best fixed action by composite ==")
    print(best.sort_values("domain").to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", required=True, help="freqduet_ablation_per_seed.csv from aggregate-only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    per_seed_path = Path(args.per_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(per_seed_path)
    metrics = metric_columns(raw, [m.strip() for m in args.metrics.split(",") if m.strip()])
    prepared = with_overall(prepare(raw), metrics)

    group_summary = summarize_groups(prepared, metrics, n_boot=args.n_boot)
    vs_zero = paired_vs_zero(prepared, metrics, n_boot=args.n_boot)
    mode_deltas = paired_modes(prepared, metrics, n_boot=args.n_boot)
    best = best_by_domain(prepared, group_summary, n_boot=args.n_boot)

    group_summary.to_csv(out_dir / "action_counterfactual_group_summary.csv", index=False)
    vs_zero.to_csv(out_dir / "action_counterfactual_delta_vs_zero.csv", index=False)
    mode_deltas.to_csv(out_dir / "action_counterfactual_mode_deltas.csv", index=False)
    best.to_csv(out_dir / "action_counterfactual_best_by_domain.csv", index=False)
    (out_dir / "action_counterfactual_summary.json").write_text(
        json.dumps(
            {
                "per_seed": str(per_seed_path),
                "metrics": metrics,
                "n_boot": args.n_boot,
                "rows": {
                    "group_summary": int(len(group_summary)),
                    "delta_vs_zero": int(len(vs_zero)),
                    "mode_deltas": int(len(mode_deltas)),
                    "best_by_domain": int(len(best)),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_compact(group_summary, vs_zero, mode_deltas, best)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
