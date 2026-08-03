"""Machine-checkable frozen Freq-HRL protocol specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPEC_VERSION = "freq_hrl_v2_spec_2026_08_03"


REQUIRED_CLAIM_IDS = tuple(f"C{i}" for i in range(1, 10))

ALLOWED_CLAIM_STATUSES = (
    "supported",
    "partial",
    "not_supported",
    "missing",
)


REQUIRED_FREQUENCY_FEATURES = (
    "timestamp",
    "x_low",
    "x_low_forecast",
    "x_low_uncertainty",
    "x_mid",
    "x_high",
    "x_high_energy",
    "x_high_persistence",
    "shock_age",
)


UPPER_REQUIRED_KEYS = (
    "z_upper",
    "x_low",
    "x_low_forecast",
    "x_low_uncertainty",
    "x_high_energy",
    "x_high_persistence",
    "promotion",
    "leakage_feedback",
)


LOWER_REQUIRED_KEYS = (
    "z_lower",
    "current_plan",
    "x_high",
    "x_mid",
    "shock_age",
)


UPPER_FORBIDDEN_KEYS = (
    "x_high",
    "x_high_sequence",
    "x_high_raw_sequence",
    "x_high_local_station_vector",
    "raw_high_frequency",
    "future_high",
)


LOWER_FORBIDDEN_KEYS = (
    "x_low_forecast",
    "x_low_forecast_full",
    "x_low_forecast_horizon",
    "future_low",
    "high_level_value",
)


CAUSAL_TIMESTAMP_KEYS = (
    "max_observed_timestamp",
    "observed_until",
    "window_end",
    "source_max_t",
)


@dataclass(frozen=True)
class FrozenFreqHRLSpec:
    """Frozen protocol contracts used by docs, tests, and adapters."""

    version: str = SPEC_VERSION
    required_frequency_features: tuple[str, ...] = REQUIRED_FREQUENCY_FEATURES
    upper_required_keys: tuple[str, ...] = UPPER_REQUIRED_KEYS
    lower_required_keys: tuple[str, ...] = LOWER_REQUIRED_KEYS
    upper_forbidden_keys: tuple[str, ...] = UPPER_FORBIDDEN_KEYS
    lower_forbidden_keys: tuple[str, ...] = LOWER_FORBIDDEN_KEYS
    required_claim_ids: tuple[str, ...] = REQUIRED_CLAIM_IDS
    allowed_claim_statuses: tuple[str, ...] = ALLOWED_CLAIM_STATUSES

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "required_frequency_features": list(self.required_frequency_features),
            "upper_required_keys": list(self.upper_required_keys),
            "lower_required_keys": list(self.lower_required_keys),
            "upper_forbidden_keys": list(self.upper_forbidden_keys),
            "lower_forbidden_keys": list(self.lower_forbidden_keys),
            "required_claim_ids": list(self.required_claim_ids),
            "allowed_claim_statuses": list(self.allowed_claim_statuses),
        }


def default_spec() -> FrozenFreqHRLSpec:
    return FrozenFreqHRLSpec()


def _missing_keys(data: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in data]


def _as_numeric_array(value: Any, key: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if arr.size == 0:
        raise ValueError(f"{key} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} must be finite")
    return arr


def _timestamp(value: Any, key: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a numeric timestamp") from exc
    if not np.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def validate_frequency_features(
    features: Mapping[str, Any],
    *,
    current_time: float | None = None,
    spec: FrozenFreqHRLSpec | None = None,
) -> dict[str, Any]:
    """Validate causal frequency features at a policy boundary."""
    spec = spec or default_spec()
    missing = _missing_keys(features, spec.required_frequency_features)
    if missing:
        raise ValueError(f"frequency features missing keys: {missing}")
    t = _timestamp(features["timestamp"], "timestamp")
    if current_time is not None and t > float(current_time) + 1e-9:
        raise ValueError("frequency feature timestamp exceeds current_time")
    for key in spec.required_frequency_features:
        if key == "timestamp":
            continue
        _as_numeric_array(features[key], key)
    metadata = features.get("metadata", {})
    if isinstance(metadata, Mapping):
        for key in CAUSAL_TIMESTAMP_KEYS:
            if key in metadata:
                observed_until = _timestamp(metadata[key], f"metadata.{key}")
                limit = t if current_time is None else min(t, float(current_time))
                if observed_until > limit + 1e-9:
                    raise ValueError(
                        f"frequency features violate causality: metadata.{key}={observed_until} > {limit}"
                    )
    return {"status": "supported", "timestamp": t, "version": spec.version}


def validate_upper_policy_state(
    state: Mapping[str, Any],
    *,
    spec: FrozenFreqHRLSpec | None = None,
) -> dict[str, Any]:
    """Validate that an upper policy state follows the frozen routing contract."""
    spec = spec or default_spec()
    missing = _missing_keys(state, spec.upper_required_keys)
    if missing:
        raise ValueError(f"upper policy state missing keys: {missing}")
    forbidden = [key for key in spec.upper_forbidden_keys if key in state]
    if forbidden:
        raise ValueError(f"upper policy state contains forbidden high-frequency keys: {forbidden}")
    for key in ("x_low", "x_low_forecast", "x_low_uncertainty", "x_high_energy", "x_high_persistence"):
        _as_numeric_array(state[key], key)
    return {"status": "supported", "version": spec.version}


def validate_lower_policy_state(
    state: Mapping[str, Any],
    *,
    spec: FrozenFreqHRLSpec | None = None,
) -> dict[str, Any]:
    """Validate that a lower policy state follows the frozen routing contract."""
    spec = spec or default_spec()
    missing = _missing_keys(state, spec.lower_required_keys)
    if missing:
        raise ValueError(f"lower policy state missing keys: {missing}")
    forbidden = [key for key in spec.lower_forbidden_keys if key in state]
    if forbidden:
        raise ValueError(f"lower policy state contains forbidden low-frequency planning keys: {forbidden}")
    for key in ("x_high", "x_mid", "shock_age"):
        _as_numeric_array(state[key], key)
    return {"status": "supported", "version": spec.version}


def validate_claim_freeze(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: FrozenFreqHRLSpec | None = None,
) -> dict[str, Any]:
    """Validate claim identity and status without forcing an all-green ledger."""
    spec = spec or default_spec()
    claim_ids = [str(row.get("claim_id", row.get("id", ""))) for row in rows]
    duplicates = sorted({claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1})
    by_id = {claim_id: row for claim_id, row in zip(claim_ids, rows)}
    missing = [claim_id for claim_id in spec.required_claim_ids if claim_id not in by_id]
    invalid_status = [
        claim_id for claim_id in spec.required_claim_ids
        if claim_id in by_id
        and str(by_id[claim_id].get("status", "")) not in spec.allowed_claim_statuses
    ]
    if missing or duplicates or invalid_status:
        raise ValueError(
            "claim freeze failed: "
            f"missing={missing} duplicates={duplicates} invalid_status={invalid_status}"
        )
    counts = {
        status: sum(str(by_id[claim_id].get("status", "")) == status for claim_id in spec.required_claim_ids)
        for status in spec.allowed_claim_statuses
    }
    return {
        "status": "valid",
        "version": spec.version,
        "claims": len(rows),
        "required_claims": len(spec.required_claim_ids),
        "status_counts": counts,
    }


def validate_shared_core_paths(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_root: Path = Path("."),
    spec: FrozenFreqHRLSpec | None = None,
) -> dict[str, Any]:
    """Validate that shared-core audit rows point to real artifacts."""
    spec = spec or default_spec()
    missing = []
    for row in rows:
        path = str(row.get("path", ""))
        if path and not (source_root / path).exists():
            missing.append(path)
    if missing:
        raise ValueError(f"shared-core audit paths missing: {missing}")
    return {
        "status": "supported",
        "version": spec.version,
        "shared_core_rows": len(rows),
    }
