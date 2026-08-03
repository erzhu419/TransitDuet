"""Evidence provenance rules shared by experiment and manuscript pipelines."""

from __future__ import annotations

from typing import Any, Mapping


OBSERVED_EVIDENCE = "observed_environment_outcome"
PROJECTION_EVIDENCE = "counterfactual_projection"
DIAGNOSTIC_EVIDENCE = "mechanism_diagnostic"

_PROJECTION_TOKENS = (
    "adjustment",
    "adjusted",
    "counterfactual",
    "projected_",
    "projection",
)


def metric_evidence_class(metric: str) -> str:
    name = str(metric).lower()
    if any(token in name for token in _PROJECTION_TOKENS):
        return PROJECTION_EVIDENCE
    if name.startswith("shared_ppo_") or name.startswith("service_adjustment_"):
        return DIAGNOSTIC_EVIDENCE
    return OBSERVED_EVIDENCE


def annotate_check(
    row: Mapping[str, Any],
    *,
    evidence_class: str | None = None,
    headline_eligible: bool | None = None,
) -> dict[str, Any]:
    """Attach provenance without changing the exploratory statistical status."""
    out = dict(row)
    evidence = str(evidence_class or metric_evidence_class(str(out.get("metric", ""))))
    if headline_eligible is None:
        headline_eligible = evidence == OBSERVED_EVIDENCE
    out["evidence_class"] = evidence
    out["headline_eligible"] = bool(headline_eligible)
    out["headline_status"] = (
        str(out.get("status", "missing")) if headline_eligible else "ineligible"
    )
    return out


def is_headline_eligible(row: Mapping[str, Any]) -> bool:
    if "headline_eligible" in row:
        return bool(row.get("headline_eligible"))
    return metric_evidence_class(str(row.get("metric", ""))) == OBSERVED_EVIDENCE


def headline_status(row: Mapping[str, Any]) -> str:
    return str(row.get("status", "missing")) if is_headline_eligible(row) else "ineligible"


def observed_value(
    row: Mapping[str, Any],
    canonical: str,
    *legacy_raw_aliases: str,
    default: float = 0.0,
) -> float:
    """Read an observed value, preferring explicit raw fields from v1 artifacts."""
    for key in legacy_raw_aliases:
        if key in row:
            return float(row[key])
    return float(row.get(canonical, default))
