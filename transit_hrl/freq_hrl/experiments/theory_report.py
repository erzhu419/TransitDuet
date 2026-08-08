"""Write a compact report from the canonical Freq-HRL formal appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .theory_appendix import (
    FORMAL_SCOPE_VERSION,
    build_formal_statement_rows,
    build_numeric_examples,
    build_reporting_rules,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not checks:
        return {"n_checks": 0}
    n_common = [float(row.get("n_common", 0) or 0) for row in checks]
    supported = [
        row for row in checks
        if str(row.get("status")) in {"supported", "positive_mixed"}
    ]
    return {
        "n_checks": len(checks),
        "supported_or_positive": len(supported),
        "min_pairs": int(min(n_common)),
        "median_pairs": float(np.median(n_common)),
        "max_pairs": int(max(n_common)),
    }


def write_report(
    path: Path,
    formal_statements: list[dict[str, Any]],
    reporting_rules: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    lines = [
        "# Freq-HRL Formal-Scope Diagnostics",
        "",
        "This report separates proved scope-limited statements from empirical "
        "diagnostics and reporting approximations. It is not a universal RL "
        "convergence claim.",
        "",
        "## Statistical Coverage",
        "",
        f"- checks: {stats.get('n_checks', 0)}",
        f"- supported or positive-mixed: {stats.get('supported_or_positive', 0)}",
        f"- paired counts: min={stats.get('min_pairs', 0)}, "
        f"median={stats.get('median_pairs', 0)}, max={stats.get('max_pairs', 0)}",
        "",
        "## Formal Statements",
        "",
    ]
    for row in formal_statements:
        lines.extend([
            f"### {row['id']} ({row['kind']}): {row['title']}",
            "",
            f"Statement: {row['statement']}",
            "",
            "Assumptions:",
        ])
        lines.extend(f"- {assumption}" for assumption in row["assumptions"])
        lines.extend([
            "",
            f"Proof: {row['proof']}",
            "",
            f"Limitation: {row['limitation']}",
            "",
        ])
    lines.extend(["## Reporting Rules", ""])
    for row in reporting_rules:
        lines.extend([
            f"### {row['id']} ({row['kind']}): {row['title']}",
            "",
            f"Rule: {row['statement']}",
            "",
            f"Limitation: {row['limitation']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_theory_report(
    output_dir: Path,
    paper_diagnostics_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper = read_json(paper_diagnostics_path)
    checks = paper.get("statistical_checks", []) if isinstance(paper, dict) else []
    stats = summarize_checks(checks)
    examples = build_numeric_examples()
    formal_statements = build_formal_statement_rows(examples)
    reporting_rules = build_reporting_rules(examples)
    payload = {
        "schema_version": FORMAL_SCOPE_VERSION,
        "proof_verification_status": "internal_scope_audit_pass",
        "independent_proof_verification": False,
        "formal_statements": formal_statements,
        "reporting_rules": reporting_rules,
        "statistical_coverage": stats,
        "paper_diagnostics_path": str(paper_diagnostics_path),
    }
    with (output_dir / "theory_claims.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_report(
        output_dir / "theory_report.md",
        formal_statements,
        reporting_rules,
        stats,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-diagnostics",
        type=Path,
        default=Path(
            "transit_hrl/results/freq_hrl_paper_diagnostics/claim_matrix.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/freq_hrl_theory_diagnostics"),
    )
    args = parser.parse_args()
    payload = run_theory_report(args.output_dir, args.paper_diagnostics)
    coverage = payload["statistical_coverage"]
    print(f"wrote {args.output_dir}")
    print(
        "theory_report "
        f"formal_statements={len(payload['formal_statements'])} "
        f"checks={coverage.get('n_checks', 0)} "
        f"median_pairs={coverage.get('median_pairs', 0)}"
    )


if __name__ == "__main__":
    main()
