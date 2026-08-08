"""Shared fail-fast gate for paper artifacts that are not submission-ready."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def read_submission_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"paper manifest must be a mapping: {path}")
    return payload


def require_submission_ready(
    manifest: Mapping[str, Any], *, allow_historical: bool = False
) -> None:
    status = str(manifest.get("submission_status", "ready")).strip().lower()
    if status.startswith("hold") and not allow_historical:
        protocol = str(manifest.get("active_protocol", "unknown"))
        raise RuntimeError(
            "paper artifact generation is blocked by submission_status="
            f"{status!r}; active protocol is {protocol!r}. Pass "
            "--allow-historical only to reproduce the explicitly historical "
            "package."
        )
