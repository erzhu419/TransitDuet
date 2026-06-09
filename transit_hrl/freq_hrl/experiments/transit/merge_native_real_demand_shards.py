"""Merge native real-demand Transit control validation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.transit.native_real_demand_control_validation import (
    VARIANTS,
    _row_from_payload,
    paired_checks,
    write_outputs,
)


def _read_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_native_real_demand_shards(
    input_dirs: list[Path],
    output_dir: Path,
    min_pairs: int = 10,
) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    metadata_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    config_paths: list[str] = []
    control_profiles: set[str] = set()
    sources: set[str] = set()
    episodes = 0
    for input_dir in input_dirs:
        summary_path = Path(input_dir) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        payload = _read_payload(summary_path)
        if payload.get("config_path"):
            config_paths.append(str(payload["config_path"]))
        if payload.get("control_profile"):
            control_profiles.add(str(payload["control_profile"]))
        episodes = max(episodes, int(payload.get("episodes", 0)))
        sources.update(str(source) for source in payload.get("sources", []))
        for meta in payload.get("metadata", []):
            key = (str(meta.get("source", "")), str(meta.get("boundary", "")))
            metadata_by_key[key] = dict(meta)
        for row in payload.get("rows", []):
            key = (
                str(row.get("source", "")),
                int(row.get("seed", 0)),
                str(row.get("variant", "")),
            )
            rows_by_key[key] = dict(row)
        for key, compact in (payload.get("payloads", {}) or {}).items():
            payloads[str(key)] = compact
            try:
                source, seed_text, variant = str(key).split(":", 2)
                row_key = (source, int(seed_text), variant)
            except ValueError:
                continue
            rows_by_key[row_key] = _row_from_payload(
                source=source,
                seed=int(seed_text),
                variant=variant,
                payload=compact,
            )
    rows = [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (
                item[0],
                item[1],
                list(VARIANTS).index(item[2]) if item[2] in VARIANTS else 999,
            ),
        )
    ]
    if not rows:
        raise ValueError("no native real-demand rows found in input shard summaries")
    checks = paired_checks(rows, min_pairs=int(min_pairs))
    seeds = sorted({int(row["seed"]) for row in rows})
    merged = {
        "metadata": list(metadata_by_key.values()),
        "config_path": config_paths[0] if config_paths else "",
        "merged_config_paths": sorted(set(config_paths)),
        "sources": sorted(sources),
        "seeds": seeds,
        "episodes": int(episodes),
        "min_pairs": int(min_pairs),
        "control_profiles": sorted(control_profiles),
        "variants": list(VARIANTS.keys()),
        "rows": rows,
        "paired_checks": checks,
        "payloads": payloads,
        "merged_input_dirs": [str(path) for path in input_dirs],
        "boundary": "merged native simulator passenger loop with public AFC/APC profile mapping",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_native_real_demand_control_merged"),
    )
    parser.add_argument("--min-pairs", type=int, default=10)
    args = parser.parse_args()
    payload = merge_native_real_demand_shards(
        input_dirs=list(args.input_dirs),
        output_dir=args.output_dir,
        min_pairs=int(args.min_pairs),
    )
    score = next(
        row for row in payload["paired_checks"]
        if row["metric"] == "control_score"
    )
    wait = next(
        row for row in payload["paired_checks"]
        if row["metric"] == "native_avg_board_wait_min"
    )
    print(
        "DONE merged native_real_demand "
        f"rows={len(payload['rows'])} seeds={len(payload['seeds'])} "
        f"score_delta={score['delta_mean']:+.4f} score_status={score['status']} "
        f"wait_delta={wait['delta_mean']:+.4f} wait_status={wait['status']}"
    )


if __name__ == "__main__":
    main()
