"""Phase-0 logging-only trace format and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PHASE0_SCHEMA_VERSION = "freq_hrl.phase0.v1"
PHASE0_REQUIRED_FIELDS = (
    "schema_version",
    "t",
    "domain",
    "entity_id",
    "x_raw",
    "x_bin",
    "z_t",
    "a_U",
    "a_L",
    "plan_curve",
    "action_effects",
    "reward",
)


def phase0_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): phase0_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [phase0_to_jsonable(v) for v in value]
    return value


class Phase0TraceLogger:
    """JSONL writer for causal logging-only audits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.n = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Phase0TraceLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, record: Mapping[str, Any]) -> None:
        out = dict(record)
        out.setdefault("schema_version", PHASE0_SCHEMA_VERSION)
        missing = [field for field in PHASE0_REQUIRED_FIELDS if field not in out]
        if missing:
            raise ValueError(f"phase0 record missing required fields: {missing}")
        self._fh.write(json.dumps(phase0_to_jsonable(out), sort_keys=True) + "\n")
        self.n += 1


def load_phase0_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_phase0_record_schema(record: Mapping[str, Any]) -> None:
    missing = [field for field in PHASE0_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"phase0 record missing required fields: {missing}")
    if record.get("schema_version") != PHASE0_SCHEMA_VERSION:
        raise ValueError(f"unknown phase0 schema: {record.get('schema_version')}")
