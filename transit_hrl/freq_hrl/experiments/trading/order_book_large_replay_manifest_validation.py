"""Manifest-driven large L2/L3 order-book replay validation.

The existing L2 matching and L3 FIFO replay runners support CSV input. This
wrapper adds a dataset manifest so large venue/symbol/date collections can be
validated without hard-coding file lists in experiment commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from freq_hrl.experiments.trading.order_book_data import ORDER_BOOK_ENCODERS
from freq_hrl.experiments.trading import order_book_l3_replay_validation as l3_replay
from freq_hrl.experiments.trading import order_book_matching_validation as l2_matching


@dataclass(frozen=True)
class ManifestEntry:
    kind: str
    path: Path
    venue: str = ""
    symbol: str = ""
    session: str = ""
    source_id: str = ""
    source_type: str = ""


def _as_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("datasets"), list):
            return [dict(item) for item in raw["datasets"]]
        if isinstance(raw.get("entries"), list):
            return [dict(item) for item in raw["entries"]]
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    raise ValueError("manifest must be a list or a dict with `datasets`/`entries`")


def load_manifest(path: Path) -> list[ManifestEntry]:
    base = Path(path).parent
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        records = _as_entries(raw)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            records = [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]
    else:
        raise ValueError(f"unsupported manifest suffix: {path}")

    entries: list[ManifestEntry] = []
    for idx, record in enumerate(records):
        kind = str(record.get("kind") or record.get("type") or record.get("book_type") or "").strip().lower()
        if kind not in {"l2", "l3"}:
            raise ValueError(f"manifest row {idx} has unsupported kind: {kind!r}")
        raw_path = Path(str(record.get("path") or record.get("csv") or record.get("file") or ""))
        if not str(raw_path):
            raise ValueError(f"manifest row {idx} is missing path")
        resolved = raw_path if raw_path.is_absolute() else base / raw_path
        entries.append(ManifestEntry(
            kind=kind,
            path=resolved.resolve(),
            venue=str(record.get("venue", "")),
            symbol=str(record.get("symbol", "")),
            session=str(record.get("session") or record.get("date") or ""),
            source_id=str(record.get("source_id") or f"{kind}_{idx}"),
            source_type=str(
                record.get("source_type")
                or record.get("data_type")
                or record.get("quality")
                or ""
            ).strip().lower(),
        ))
    return entries


def _filter_existing(
    entries: list[ManifestEntry],
    *,
    allow_missing: bool,
    max_files: int,
) -> tuple[list[ManifestEntry], list[dict[str, Any]]]:
    kept: list[ManifestEntry] = []
    missing: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.path.exists():
            item = {
                "kind": entry.kind,
                "path": str(entry.path),
                "venue": entry.venue,
                "symbol": entry.symbol,
                "session": entry.session,
                "source_id": entry.source_id,
                "source_type": entry.source_type,
            }
            if not allow_missing:
                raise FileNotFoundError(f"manifest input missing: {entry.path}")
            missing.append(item)
            continue
        kept.append(entry)
        if max_files > 0 and len(kept) >= max_files:
            break
    return kept, missing


def _metadata_by_path(entries: list[ManifestEntry]) -> dict[str, dict[str, str]]:
    return {
        str(entry.path): {
            "venue": entry.venue,
            "symbol": entry.symbol,
            "session": entry.session,
            "source_id": entry.source_id,
            "source_type": entry.source_type,
        }
        for entry in entries
    }


def _is_real_or_venue_grade(entry: ManifestEntry) -> bool:
    source_type = str(entry.source_type).lower().replace("-", "_")
    if source_type in {"real", "venue", "venue_grade", "exchange", "production"}:
        return True
    source_id = str(entry.source_id).lower()
    path_text = str(entry.path).lower()
    if any(token in source_id or token in path_text for token in ("synthetic", "fixture", "toy", "sample")):
        return False
    return False


def _enrich_rows(rows: list[dict[str, Any]], *, kind: str, entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    metadata = _metadata_by_path(entries)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        source = str(item.get("source", ""))
        item["book_kind"] = str(kind)
        item.update(metadata.get(source, {}))
        enriched.append(item)
    return enriched


def _prefix_checks(checks: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in checks:
        item = dict(row)
        item["check"] = f"{prefix}_{item.get('check', 'unknown')}"
        item["book_kind"] = str(prefix)
        out.append(item)
    return out


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_manifest_validation(
    output_dir: Path,
    *,
    manifest: Path,
    methods: list[str],
    steps: int,
    levels: int,
    latency_bins: list[int],
    execution_modes: list[str],
    queue_ahead_fraction: float,
    min_pairs: int,
    max_files: int = 0,
    allow_missing: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_raw = load_manifest(manifest)
    entries, missing = _filter_existing(
        entries_raw,
        allow_missing=bool(allow_missing),
        max_files=int(max_files),
    )
    l2_entries = [entry for entry in entries if entry.kind == "l2"]
    l3_entries = [entry for entry in entries if entry.kind == "l3"]

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}
    if l2_entries:
        l2_payload = l2_matching.run_validation(
            output_dir / "l2_matching",
            seeds=[],
            csv_files=[entry.path for entry in l2_entries],
            methods=list(methods),
            latency_bins=list(latency_bins),
            execution_modes=list(execution_modes),
            queue_ahead_fraction=float(queue_ahead_fraction),
            steps=int(steps),
            levels=int(levels),
            min_pairs=int(min_pairs),
        )
        l2_rows = _enrich_rows(l2_payload["summary"], kind="l2", entries=l2_entries)
        rows.extend(l2_rows)
        checks.extend(_prefix_checks(l2_payload["paired_checks"], "l2"))
        sections["l2"] = {
            "files": [str(entry.path) for entry in l2_entries],
            "rows": len(l2_rows),
            "paired_checks": l2_payload["paired_checks"],
        }
    if l3_entries:
        l3_payload = l3_replay.run_validation(
            output_dir / "l3_replay",
            seeds=[],
            csv_files=[entry.path for entry in l3_entries],
            methods=list(methods),
            steps=int(steps),
            levels=int(levels),
            min_pairs=int(min_pairs),
        )
        l3_rows = _enrich_rows(l3_payload["summary"], kind="l3", entries=l3_entries)
        rows.extend(l3_rows)
        checks.extend(_prefix_checks(l3_payload["paired_checks"], "l3"))
        sections["l3"] = {
            "files": [str(entry.path) for entry in l3_entries],
            "rows": len(l3_rows),
            "paired_checks": l3_payload["paired_checks"],
        }
    if not rows:
        raise ValueError("manifest produced no valid L2/L3 validation rows")

    coverage = {
        "manifest_entries": len(entries_raw),
        "used_entries": len(entries),
        "missing_entries": len(missing),
        "l2_files": len(l2_entries),
        "l3_files": len(l3_entries),
        "real_l2_files": sum(1 for entry in l2_entries if _is_real_or_venue_grade(entry)),
        "real_l3_files": sum(1 for entry in l3_entries if _is_real_or_venue_grade(entry)),
        "fixture_l2_files": sum(1 for entry in l2_entries if not _is_real_or_venue_grade(entry)),
        "fixture_l3_files": sum(1 for entry in l3_entries if not _is_real_or_venue_grade(entry)),
        "methods": list(methods),
        "steps": int(steps),
        "levels": int(levels),
        "latency_bins": list(latency_bins),
        "execution_modes": list(execution_modes),
    }
    real_sessions = {
        (
            entry.venue,
            entry.symbol,
            entry.session,
            entry.kind,
        )
        for entry in entries
        if _is_real_or_venue_grade(entry)
    }
    if coverage["real_l2_files"] > 0 and coverage["real_l3_files"] > 0:
        source_quality_status = "venue_grade_ready"
    elif coverage["l2_files"] > 0 and coverage["l3_files"] > 0:
        source_quality_status = "mechanism_only"
    else:
        source_quality_status = "incomplete"
    coverage.update({
        "real_or_venue_grade_sessions": len(real_sessions),
        "source_quality_status": source_quality_status,
    })
    payload = {
        "manifest": str(manifest),
        "coverage": coverage,
        "missing": missing,
        "sections": sections,
        "summary": rows,
        "paired_checks": checks,
        "boundary": (
            "Manifest-driven real/fixture L2/L3 replay. L2 uses market/passive "
            "matching with a best-level queue-priority proxy; L3 uses FIFO add/"
            "cancel/trade event replay for agent passive orders. Venue-grade "
            "claims require real exchange feeds in the manifest."
        ),
    }
    _write_table(output_dir / "per_eval.csv", rows)
    _write_table(output_dir / "paired_checks.csv", checks)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Order-Book Large Replay Manifest Validation",
        "",
        f"- manifest entries: `{coverage['manifest_entries']}`",
        f"- used entries: `{coverage['used_entries']}`",
        f"- L2 files: `{coverage['l2_files']}`",
        f"- L3 files: `{coverage['l3_files']}`",
        f"- real/venue-grade L2 files: `{coverage['real_l2_files']}`",
        f"- real/venue-grade L3 files: `{coverage['real_l3_files']}`",
        f"- source quality: `{coverage['source_quality_status']}`",
        f"- missing entries: `{coverage['missing_entries']}`",
        f"- boundary: {payload['boundary']}",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} | {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=ORDER_BOOK_ENCODERS, default=list(ORDER_BOOK_ENCODERS))
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--latency-bins", type=int, nargs="+", default=[0, 2, 5])
    parser.add_argument("--execution-modes", nargs="+", default=["market", "passive_queue"])
    parser.add_argument("--queue-ahead-fraction", type=float, default=0.50)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_order_book_large_replay_manifest"),
    )
    args = parser.parse_args()
    payload = run_manifest_validation(
        args.output_dir,
        manifest=args.manifest,
        methods=list(args.methods),
        steps=int(args.steps),
        levels=int(args.levels),
        latency_bins=list(args.latency_bins),
        execution_modes=list(args.execution_modes),
        queue_ahead_fraction=float(args.queue_ahead_fraction),
        min_pairs=int(args.min_pairs),
        max_files=int(args.max_files),
        allow_missing=bool(args.allow_missing),
    )
    coverage = payload["coverage"]
    print(
        "order_book_large_replay "
        f"used={coverage['used_entries']} l2={coverage['l2_files']} l3={coverage['l3_files']}"
    )


if __name__ == "__main__":
    main()
