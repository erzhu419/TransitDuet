#!/usr/bin/env python3
"""Export executable FreqDuet configs from matched counterfactual rollout labels.

The counterfactual audit produces per-domain, per-seed rollout costs for several
executable candidates. This script turns those labels into an explicit domain
policy and, optionally, YAML configs that can be submitted to the scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
DEFAULT_ALLOWED_METHODS = (
    "main",
    "fixed",
    "odgate",
    "safe",
    "discctx",
    "discctx_hfgate",
    "odgate_fixedrule",
)
DOMAIN_PREFIX = {
    "terminal": "F_freqduet_terminal_main",
    "highnoise": "F_freqduet_gen_highnoise_main",
    "odshift": "F_freqduet_gen_odshift_main",
    "rushshift": "F_freqduet_gen_rushshift_main",
}
METHOD_EXTENDS_SUFFIX = {
    "main": "_hiro",
    "odgate": "_headwayplanner_odgate_hiro",
    "safe": "_headwayplanner_safe_hiro",
    "discctx": "_headwayplanner_discctx_hiro",
    "discctx_hfgate": "_headwayplanner_discctx_hfgate_hiro",
    "odgate_fixedrule": "_headwayplanner_odgate_fixedrule_hiro",
}
FIXED_EXPERT_BLOCK = """fixed_expert_selector:
  enable: true
  strict_headway_s: 360.0
  reset_env_rng: true
  start_ep: 0
  count_start: selector_start
  deterministic_rule:
    enable: true
    default: fixed
"""


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def parse_methods(text: str) -> list[str]:
    methods = [part.strip() for part in text.split(",") if part.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("method list is empty")
    unsupported = sorted(set(methods) - set(DEFAULT_ALLOWED_METHODS))
    if unsupported:
        raise argparse.ArgumentTypeError(
            "unsupported executable candidate methods: " + ", ".join(unsupported)
        )
    return methods


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def read_rows(path: Path, metric: str, allowed_methods: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"counterfactual rows file does not exist: {path}")
    df = pd.read_csv(path)
    required = {"domain", "seed", "candidate_method", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"counterfactual rows missing columns: {missing}")
    df = df[df["domain"].isin(DOMAINS)].copy()
    df = df[df["candidate_method"].isin(allowed_methods)].copy()
    if df.empty:
        raise SystemExit("no rows left after domain/method filtering")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric])].copy()
    dup = df.duplicated(["domain", "seed", "candidate_method"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["domain", "seed", "candidate_method"]].head(20)
        raise SystemExit(
            "duplicate domain/seed/candidate rows:\n"
            f"{bad.to_string(index=False)}"
        )
    return df


def wide_by_domain(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for domain, group in df.groupby("domain"):
        wide = group.pivot(index="seed", columns="candidate_method", values=metric)
        wide = wide.dropna(axis=1, how="all")
        out[str(domain)] = wide
    return out


def choose_domain_mean(
    wide: pd.DataFrame,
    allowed_methods: list[str],
    fallback_method: str,
    n_boot: int,
    domain: str,
) -> dict[str, object]:
    available = [method for method in allowed_methods if method in wide.columns]
    means = {method: float(wide[method].mean()) for method in available}
    selected = min(available, key=lambda method: means[method])
    return summarize_choice(
        wide=wide,
        selected=selected,
        fallback_method=fallback_method,
        n_boot=n_boot,
        domain=domain,
        selector="domain_mean",
        reason="lowest matched-seed domain mean among executable candidates",
    )


def choose_conservative_vs_fixed(
    wide: pd.DataFrame,
    allowed_methods: list[str],
    fallback_method: str,
    min_improvement: float,
    n_boot: int,
    domain: str,
) -> dict[str, object]:
    if fallback_method not in wide.columns:
        raise SystemExit(f"fallback method {fallback_method!r} missing for {domain}")
    candidates = [
        method
        for method in allowed_methods
        if method in wide.columns and method != fallback_method
    ]
    if not candidates:
        return summarize_choice(
            wide=wide,
            selected=fallback_method,
            fallback_method=fallback_method,
            n_boot=n_boot,
            domain=domain,
            selector="conservative_vs_fixed",
            reason="no non-fallback executable candidates available",
        )
    stats = []
    for method in candidates:
        paired = wide[[method, fallback_method]].dropna()
        if paired.empty:
            continue
        delta = (paired[method] - paired[fallback_method]).to_numpy(dtype=np.float64)
        lo, hi = bootstrap_ci(
            delta,
            n_boot=n_boot,
            seed=stable_seed("conservative_vs_fixed", domain, method, fallback_method),
        )
        stats.append((method, float(delta.mean()), lo, hi))
    passed = [
        item
        for item in stats
        if item[1] <= -float(min_improvement) and item[3] < 0.0
    ]
    if passed:
        selected, mean_delta, _lo, hi = min(passed, key=lambda item: item[1])
        reason = (
            f"best candidate with mean delta <= -{min_improvement:g} and "
            f"bootstrap CI upper < 0 versus {fallback_method}; "
            f"selected delta {mean_delta:+.4f}, ci_hi {hi:+.4f}"
        )
    else:
        selected = fallback_method
        reason = (
            f"no candidate cleared mean delta <= -{min_improvement:g} and "
            f"bootstrap CI upper < 0 versus {fallback_method}"
        )
    return summarize_choice(
        wide=wide,
        selected=selected,
        fallback_method=fallback_method,
        n_boot=n_boot,
        domain=domain,
        selector="conservative_vs_fixed",
        reason=reason,
    )


def summarize_choice(
    wide: pd.DataFrame,
    selected: str,
    fallback_method: str,
    n_boot: int,
    domain: str,
    selector: str,
    reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "selector": selector,
        "domain": domain,
        "selected_method": selected,
        "selected_config": config_name(domain, selected, selector),
        "n_seeds": int(wide[selected].dropna().shape[0]),
        "selected_mean": float(wide[selected].mean()),
        "reason": reason,
    }
    means = {method: float(wide[method].mean()) for method in wide.columns}
    row["candidate_means_json"] = json.dumps(means, sort_keys=True)
    if fallback_method in wide.columns:
        row["fallback_method"] = fallback_method
        if selected == fallback_method:
            fallback_values = wide[fallback_method].dropna()
            row["fallback_mean"] = float(fallback_values.mean())
            row["delta_vs_fallback_mean"] = 0.0
            row["delta_vs_fallback_ci95_lo"] = 0.0
            row["delta_vs_fallback_ci95_hi"] = 0.0
        else:
            paired = wide[[selected, fallback_method]].dropna()
            row["fallback_mean"] = float(paired[fallback_method].mean())
            if paired.empty:
                row["delta_vs_fallback_mean"] = math.nan
                row["delta_vs_fallback_ci95_lo"] = math.nan
                row["delta_vs_fallback_ci95_hi"] = math.nan
            else:
                delta = (paired[selected] - paired[fallback_method]).to_numpy(dtype=np.float64)
                lo, hi = bootstrap_ci(
                    delta,
                    n_boot=n_boot,
                    seed=stable_seed("choice", selector, domain, selected, fallback_method),
                )
                row["delta_vs_fallback_mean"] = float(delta.mean())
                row["delta_vs_fallback_ci95_lo"] = lo
                row["delta_vs_fallback_ci95_hi"] = hi
    return row


def config_name(domain: str, method: str, name_prefix: str) -> str:
    return f"{DOMAIN_PREFIX[domain]}_{name_prefix}_hiro"


def extends_name(domain: str, method: str) -> str:
    prefix = DOMAIN_PREFIX[domain]
    if method in ("fixed", "fixed_headway"):
        return f"{prefix}_hiro.yaml"
    suffix = METHOD_EXTENDS_SUFFIX.get(method)
    if suffix is None:
        raise SystemExit(f"cannot export executable config for method {method!r}")
    return f"{prefix}{suffix}.yaml"


def write_config(
    config_dir: Path,
    domain: str,
    method: str,
    name_prefix: str,
    selector: str,
    reason: str,
) -> Path:
    name = config_name(domain, method, name_prefix)
    path = config_dir / f"{name}.yaml"
    lines = [
        f"_extends: {extends_name(domain, method)}",
        f"_name: {name}",
        "",
        f"# Exported by export_freqduet_counterfactual_selector_configs.py.",
        f"# selector: {selector}",
        f"# selected_method: {method}",
        f"# reason: {reason}",
    ]
    if method in ("fixed", "fixed_headway"):
        lines.extend(["", FIXED_EXPERT_BLOCK.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-rows",
        type=Path,
        required=True,
        help="counterfactual_candidate_rows.csv from the value audit.",
    )
    ap.add_argument("--metric", default="composite")
    ap.add_argument(
        "--selector",
        choices=("domain_mean", "conservative_vs_fixed"),
        default="conservative_vs_fixed",
    )
    ap.add_argument(
        "--allowed-methods",
        type=parse_methods,
        default=list(DEFAULT_ALLOWED_METHODS),
        help="Comma-separated executable candidate methods.",
    )
    ap.add_argument("--fallback-method", default="fixed")
    ap.add_argument("--min-improvement", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs_freqduet"),
        help="Directory to write configs when --write-configs is set.",
    )
    ap.add_argument(
        "--name-prefix",
        default="cfvalue_conservative",
        help="Inserted into generated config names after the domain main prefix.",
    )
    ap.add_argument("--write-configs", action="store_true")
    args = ap.parse_args()

    if args.fallback_method not in args.allowed_methods:
        raise SystemExit("--fallback-method must be included in --allowed-methods")

    rows = read_rows(args.candidate_rows, args.metric, args.allowed_methods)
    by_domain = wide_by_domain(rows, args.metric)
    missing_domains = sorted(set(DOMAINS) - set(by_domain))
    if missing_domains:
        raise SystemExit("candidate rows missing domains: " + ", ".join(missing_domains))

    policy_rows = []
    for domain in DOMAINS:
        wide = by_domain[domain]
        if args.selector == "domain_mean":
            row = choose_domain_mean(
                wide,
                allowed_methods=args.allowed_methods,
                fallback_method=args.fallback_method,
                n_boot=args.n_boot,
                domain=domain,
            )
        else:
            row = choose_conservative_vs_fixed(
                wide,
                allowed_methods=args.allowed_methods,
                fallback_method=args.fallback_method,
                min_improvement=args.min_improvement,
                n_boot=args.n_boot,
                domain=domain,
            )
        policy_rows.append(row)
    for row in policy_rows:
        row["selected_config"] = config_name(
            str(row["domain"]),
            str(row["selected_method"]),
            args.name_prefix,
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(out_dir / "counterfactual_selector_policy.csv", index=False)

    generated = []
    if args.write_configs:
        args.config_dir.mkdir(parents=True, exist_ok=True)
        for row in policy_rows:
            path = write_config(
                config_dir=args.config_dir,
                domain=str(row["domain"]),
                method=str(row["selected_method"]),
                name_prefix=args.name_prefix,
                selector=args.selector,
                reason=str(row["reason"]),
            )
            generated.append(str(path))
        (out_dir / "generated_configs.json").write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        "candidate_rows": str(args.candidate_rows),
        "metric": args.metric,
        "selector": args.selector,
        "allowed_methods": args.allowed_methods,
        "fallback_method": args.fallback_method,
        "min_improvement": args.min_improvement,
        "n_boot": args.n_boot,
        "write_configs": bool(args.write_configs),
        "generated_configs": generated,
    }
    (out_dir / "counterfactual_selector_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(policy[[
        "domain",
        "selected_method",
        "selected_mean",
        "fallback_method",
        "fallback_mean",
        "delta_vs_fallback_mean",
        "delta_vs_fallback_ci95_lo",
        "delta_vs_fallback_ci95_hi",
    ]].to_string(index=False))
    if generated:
        print("generated configs:")
        for path in generated:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
