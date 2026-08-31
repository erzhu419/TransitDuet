"""Fail-closed checks for the current authoritative Freq-HRL manuscript."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANUSCRIPT = Path("transit_hrl/paper/manuscript.md")
DEFAULT_READINESS = Path("transit_hrl/paper/submission_readiness.md")
DEFAULT_BIBLIOGRAPHY = Path("transit_hrl/paper/references.bib")
DEFAULT_REGISTRY = Path("transit_hrl/evidence/authoritative_registry_v1.json")

REPORTABLE_TOKENS = {
    "mujoco_v12_responsibility_confirmatory": "MuJoCo v12",
    "mujoco_v13_behavioral_confirmatory": "MuJoCo v13",
    "mujoco_v14_29_restoration_portfolio_confirmatory": "v14.29",
    "quant_v74_matched_baseline_confirmatory": "Quant v7.4",
}

LEGACY_SOURCE_TOKENS = (
    "top_journal_unified_matrix_latest",
    "freq_hrl_paper_diagnostics",
    "freq_hrl_full_manuscript_draft_2026-06-27",
    "manuscript_figures_latest",
)

AUTHORITATIVE_FIGURE_TOKENS = (
    "authoritative_paper_figures_latest",
    "fig1_protocol_and_estimands.png",
    "fig2_mujoco_confirmatory_evidence.png",
    "fig3_quant_matched_baseline_forest.png",
    "fig_s1_development_stop_map.png",
)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _citation_keys(manuscript: str) -> set[str]:
    return set(re.findall(r"(?<![\w@])@([A-Za-z0-9_.:-]+)", manuscript))


def _bibliography_keys(bibliography: str) -> set[str]:
    return set(
        re.findall(
            r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,",
            bibliography,
            flags=re.IGNORECASE,
        )
    )


def _registry_counts(registry: dict[str, Any]) -> dict[str, int]:
    records = registry["records"]
    stages = Counter(row["evidence_stage"] for row in records)
    dispositions = Counter(row["paper_disposition"] for row in records)
    return {
        "total": len(records),
        "reportable": dispositions["positive_main_or_si"]
        + dispositions["mixed_or_negative_main_or_si"],
        "positive": dispositions["positive_main_or_si"],
        "development": stages["development"],
        "legacy": stages["legacy"],
    }


def _require_registry_counts(text: str, counts: dict[str, int], label: str) -> None:
    required = {
        f"{counts['total']} records": "total registry count",
        f"{counts['reportable']} reportable": "reportable registry count",
        f"{counts['positive']} support positive claims": "positive registry count",
        f"{counts['development']} development-only": "development registry count",
        f"{counts['legacy']} excluded legacy": "legacy registry count",
    }
    for phrase, description in required.items():
        if phrase not in text:
            raise ValueError(f"{label} is missing the current {description}: {phrase!r}")


def _check_inline_math(manuscript: str) -> None:
    for line_number, line in enumerate(manuscript.splitlines(), start=1):
        if line.strip() in {r"\[", r"\]"}:
            continue
        if len(re.findall(r"(?<!\\)\$", line)) % 2:
            raise ValueError(f"unbalanced inline math delimiter on manuscript line {line_number}")
    if r"(\square)" in manuscript:
        raise ValueError("malformed proof terminator remains in manuscript")
    if re.search(r"\([A-Za-z][^()\n]*\\[^()\n]*\\?\)", manuscript):
        raise ValueError("parenthesized LaTeX remains outside a math delimiter")


def _check_quant_table(manuscript: str) -> None:
    marker = "**Table 4. Quant v7.4 pooled matched-baseline contrasts.**"
    if marker not in manuscript:
        raise ValueError("Quant Table 4 is missing")
    table_block = manuscript.split(marker, 1)[1].split(
        "![Quant matched-baseline contrasts.]", 1
    )[0]
    rows = []
    for line in table_block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] != "Comparator":
            rows.append(cells)
    if len(rows) != 12:
        raise ValueError(f"Quant Table 4 must contain 12 contrasts, found {len(rows)}")
    keys = [(row[0], row[1]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Quant Table 4 contains a duplicate comparator-endpoint row")


def audit_current_manuscript(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manuscript_path: Path = DEFAULT_MANUSCRIPT,
    readiness_path: Path = DEFAULT_READINESS,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    manuscript = _resolve(repository_root, manuscript_path).read_text(encoding="utf-8")
    readiness = _resolve(repository_root, readiness_path).read_text(encoding="utf-8")
    bibliography = _resolve(repository_root, bibliography_path).read_text(encoding="utf-8")
    registry = json.loads(
        _resolve(repository_root, registry_path).read_text(encoding="utf-8")
    )

    citation_keys = _citation_keys(manuscript)
    bibliography_keys = _bibliography_keys(bibliography)
    missing_citations = sorted(citation_keys - bibliography_keys)
    if missing_citations:
        raise ValueError(f"manuscript citation keys missing from bibliography: {missing_citations}")

    _check_inline_math(manuscript)
    _check_quant_table(manuscript)

    missing_figures = [
        token for token in AUTHORITATIVE_FIGURE_TOKENS if token not in manuscript
    ]
    if missing_figures:
        raise ValueError(
            f"authoritative manuscript figure references are missing: {missing_figures}"
        )

    counts = _registry_counts(registry)
    _require_registry_counts(manuscript, counts, "manuscript")
    _require_registry_counts(readiness, counts, "submission readiness")

    reportable_ids = {
        row["evidence_id"]
        for row in registry["records"]
        if row["paper_disposition"]
        in {"positive_main_or_si", "mixed_or_negative_main_or_si"}
    }
    if reportable_ids != set(REPORTABLE_TOKENS):
        raise ValueError(
            "reportable registry membership changed; update the manuscript token map before submission"
        )
    missing_reportable = [
        evidence_id
        for evidence_id, token in REPORTABLE_TOKENS.items()
        if token not in manuscript
    ]
    if missing_reportable:
        raise ValueError(f"reportable evidence omitted from manuscript: {missing_reportable}")

    leaked_legacy = [token for token in LEGACY_SOURCE_TOKENS if token in manuscript]
    if leaked_legacy:
        raise ValueError(f"legacy or stale evidence source leaked into manuscript: {leaked_legacy}")

    for row in registry["records"]:
        forbidden = row["forbidden_wording"]
        if forbidden in manuscript:
            raise ValueError(f"forbidden wording copied into manuscript: {row['evidence_id']}")

    required_boundaries = (
        "No fresh validation path was accessed in\n"
        "v17.8--v18.5",
        "Further tuning on this panel would be development-set overfitting.",
        "not ready for a top-tier CS conference or journal",
    )
    combined = re.sub(r"\s+", " ", manuscript + "\n" + readiness)
    missing_boundaries = [
        phrase
        for phrase in required_boundaries
        if re.sub(r"\s+", " ", phrase) not in combined
    ]
    if missing_boundaries:
        raise ValueError(f"current development boundary is missing: {missing_boundaries}")

    snapshot_date = registry["snapshot_date"]
    if snapshot_date not in manuscript or snapshot_date not in readiness:
        raise ValueError("manuscript and readiness must name the registry snapshot date")

    return {
        "status": "pass",
        "snapshot_date": snapshot_date,
        "registry_counts": counts,
        "citation_key_count": len(citation_keys),
        "bibliography_key_count": len(bibliography_keys),
        "reportable_evidence_ids": sorted(reportable_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    print(json.dumps(audit_current_manuscript(repository_root=args.repository_root), indent=2))


if __name__ == "__main__":
    main()
