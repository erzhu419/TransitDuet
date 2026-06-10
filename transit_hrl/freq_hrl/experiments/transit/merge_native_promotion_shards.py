"""Merge native Transit promotion-replan validation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.transit.native_promotion_replan_validation import (
    VARIANTS,
    _row_from_payload,
    paired_checks,
    summarize,
    write_outputs,
)


def _read_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recover_seed_payloads(input_dir: Path) -> dict[str, dict[str, Any]]:
    recovered: dict[str, dict[str, Any]] = {}
    for variant_dir in sorted(Path(input_dir).iterdir() if Path(input_dir).exists() else []):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        by_seed: dict[str, Any] = {}
        for summary_path in sorted(variant_dir.glob("seed_*/summary.json")):
            try:
                seed = int(summary_path.parent.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            payload = _read_payload(summary_path)
            by_seed[str(seed)] = {
                "summary": payload.get("summary", {}),
                "status": payload.get("status", "missing"),
                "rows": payload.get("rows", []),
                "private_overrides": payload.get("private_overrides", {}),
            }
        if by_seed:
            recovered[variant] = by_seed
    return recovered


def merge_native_promotion_shards(
    input_dirs: list[Path],
    output_dir: Path,
    min_pairs: int = 10,
) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    variant_overrides: dict[str, Any] = {}
    variant_private_overrides: dict[str, Any] = {}
    config_paths: list[str] = []
    episodes = 0
    lower_gain = 0.0
    replay_updates = 0
    for input_dir in input_dirs:
        summary_path = Path(input_dir) / "summary.json"
        payload = _read_payload(summary_path) if summary_path.exists() else {
            "payloads": _recover_seed_payloads(Path(input_dir)),
            "rows": [],
        }
        if payload.get("config_path"):
            config_paths.append(str(payload["config_path"]))
        episodes = max(episodes, int(payload.get("episodes", 0)))
        lower_gain = max(lower_gain, float(payload.get("lower_hf_wait_action_gain_s", 0.0)))
        replay_updates = max(replay_updates, int(payload.get("offpolicy_replay_updates", 0)))
        for key, target in [
            ("variant_overrides", variant_overrides),
            ("variant_private_overrides", variant_private_overrides),
        ]:
            values = payload.get(key, {}) or {}
            if not isinstance(values, dict):
                continue
            for variant, override in values.items():
                target[str(variant)] = override
        for row in payload.get("rows", []):
            key = (str(row.get("variant")), int(row.get("seed", 0)))
            rows_by_key[key] = dict(row)
        for variant, by_seed in (payload.get("payloads", {}) or {}).items():
            if not isinstance(by_seed, dict):
                continue
            target = payloads.setdefault(str(variant), {})
            for seed_key, compact in by_seed.items():
                target[str(seed_key)] = compact
                key = (str(variant), int(seed_key))
                if key not in rows_by_key:
                    rows_by_key[key] = _row_from_payload(int(seed_key), str(variant), compact)
    rows = [
        rows_by_key[key]
        for key in sorted(rows_by_key, key=lambda item: (
            list(VARIANTS).index(item[0]) if item[0] in VARIANTS else 999,
            item[1],
        ))
    ]
    if not rows:
        raise ValueError("no native promotion rows found in input shard summaries")
    actual_variants = [
        variant for variant in VARIANTS
        if any(row.get("variant") == variant for row in rows)
    ]
    checks = []
    for treatment in (
        "native_promotion_replan",
        "native_learned_gate",
        "native_wait_aware_replan",
    ):
        if treatment in actual_variants:
            checks.extend(paired_checks(
                rows,
                min_pairs=int(min_pairs),
                treatment=treatment,
            ))
    seeds = sorted({int(row["seed"]) for row in rows})
    merged = {
        "config_path": config_paths[0] if config_paths else "",
        "merged_config_paths": sorted(set(config_paths)),
        "seeds": seeds,
        "episodes": int(episodes),
        "min_pairs": int(min_pairs),
        "lower_hf_wait_action_gain_s": float(lower_gain),
        "offpolicy_replay_updates": int(replay_updates),
        "workers": 0,
        "variants": actual_variants,
        "variant_overrides": variant_overrides,
        "variant_private_overrides": variant_private_overrides,
        "summary": summarize(rows),
        "rows": rows,
        "paired_checks": checks,
        "payloads": payloads,
        "merged_input_dirs": [str(path) for path in input_dirs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_native_promotion_replan_expanded"))
    parser.add_argument("--min-pairs", type=int, default=10)
    args = parser.parse_args()
    payload = merge_native_promotion_shards(
        input_dirs=list(args.input_dirs),
        output_dir=args.output_dir,
        min_pairs=int(args.min_pairs),
    )
    learned = next(
        (
            row for row in payload["paired_checks"]
            if row["check"] == "native_learned_gate_vs_interval_ep_reward"
        ),
        None,
    )
    print(f"merged native promotion rows={len(payload['rows'])} seeds={len(payload['seeds'])}")
    if learned is not None:
        print(
            "native_learned_gate "
            f"reward_delta={learned['delta_mean']:+.4f} "
            f"status={learned['status']}"
        )


if __name__ == "__main__":
    main()
