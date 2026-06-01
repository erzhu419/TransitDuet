"""Cross-seed confidence-interval report for Freq-HRL claim gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.statistics import finite_float, format_ci


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gate_row(row: dict[str, Any], min_pairs: int, max_p_value: float) -> dict[str, Any]:
    n_common = int(row.get("n_common", 0) or 0)
    status = str(row.get("status", "missing"))
    p_value = finite_float(row.get("sign_p_value"))
    ci_low = finite_float(row.get("improvement_ci95_low"))
    win_rate = finite_float(row.get("win_rate"))
    ci_supported = ci_low is not None and ci_low > 0.0
    p_supported = p_value is not None and p_value <= float(max_p_value)
    enough_pairs = n_common >= int(min_pairs)
    paper_ready = bool(
        enough_pairs
        and status == "supported"
        and ci_supported
        and (p_supported or n_common >= 10)
        and (win_rate is None or win_rate >= 0.70)
    )
    return {
        "check": row.get("check", ""),
        "claim": row.get("claim", ""),
        "metric": row.get("metric", ""),
        "status": status,
        "direction": row.get("direction", ""),
        "n_common": n_common,
        "delta_ci95": format_ci(row),
        "improvement_ci95_low": row.get("improvement_ci95_low", ""),
        "win_rate": row.get("win_rate", ""),
        "sign_p_value": row.get("sign_p_value", ""),
        "enough_pairs": enough_pairs,
        "ci_supported": ci_supported,
        "p_supported": p_supported,
        "paper_ready": paper_ready,
    }


def summarize(rows: list[dict[str, Any]], min_pairs: int) -> dict[str, Any]:
    if not rows:
        return {"n_checks": 0}
    n = np.asarray([int(row.get("n_common", 0) or 0) for row in rows], dtype=np.float64)
    return {
        "n_checks": len(rows),
        "n_enough_pairs": int(sum(bool(row.get("enough_pairs")) for row in rows)),
        "n_supported": int(sum(str(row.get("status")) == "supported" for row in rows)),
        "n_paper_ready": int(sum(bool(row.get("paper_ready")) for row in rows)),
        "min_pairs_gate": int(min_pairs),
        "min_n_common": int(np.min(n)),
        "median_n_common": float(np.median(n)),
        "max_n_common": int(np.max(n)),
    }


def run_cross_seed_ci_report(
    output_dir: Path,
    paper_diagnostics_path: Path,
    *,
    min_pairs: int = 5,
    max_p_value: float = 0.10,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper = read_json(paper_diagnostics_path)
    checks = paper.get("statistical_checks", []) if isinstance(paper, dict) else []
    rows = [gate_row(row, min_pairs=min_pairs, max_p_value=max_p_value) for row in checks]
    summary = summarize(rows, min_pairs=min_pairs)
    payload = {
        "paper_diagnostics_path": str(paper_diagnostics_path),
        "min_pairs": int(min_pairs),
        "max_p_value": float(max_p_value),
        "summary": summary,
        "rows": rows,
    }
    with (output_dir / "cross_seed_ci.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if rows:
        with (output_dir / "cross_seed_ci.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_report(output_dir / "report.md", payload)
    return payload


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Freq-HRL Cross-Seed CI Report",
        "",
        f"- checks: {summary.get('n_checks', 0)}",
        f"- enough paired seeds/sources: {summary.get('n_enough_pairs', 0)}",
        f"- supported checks: {summary.get('n_supported', 0)}",
        f"- paper-ready checks: {summary.get('n_paper_ready', 0)}",
        f"- n_common range: {summary.get('min_n_common', 0)} / {summary.get('median_n_common', 0)} / {summary.get('max_n_common', 0)}",
        "",
        "`paper_ready` requires supported status, enough pairs, positive improvement CI, and either sign-test p <= threshold or at least 10 pairs.",
        "",
        "| check | status | n | delta CI95 | win | p | paper ready |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check']} "
            f"| {row['status']} "
            f"| {row['n_common']} "
            f"| {row['delta_ci95']} "
            f"| {float(row['win_rate']):.2f} "
            f"| {float(row['sign_p_value']):.4f} "
            f"| {row['paper_ready']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-diagnostics",
        type=Path,
        default=Path("transit_hrl/results/freq_hrl_paper_diagnostics/claim_matrix.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/freq_hrl_cross_seed_ci"),
    )
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--max-p-value", type=float, default=0.10)
    args = parser.parse_args()
    payload = run_cross_seed_ci_report(
        output_dir=args.output_dir,
        paper_diagnostics_path=args.paper_diagnostics,
        min_pairs=int(args.min_pairs),
        max_p_value=float(args.max_p_value),
    )
    summary = payload["summary"]
    print(f"wrote {args.output_dir}")
    print(
        "cross_seed_ci "
        f"checks={summary['n_checks']} "
        f"paper_ready={summary['n_paper_ready']} "
        f"median_pairs={summary['median_n_common']}"
    )


if __name__ == "__main__":
    main()
