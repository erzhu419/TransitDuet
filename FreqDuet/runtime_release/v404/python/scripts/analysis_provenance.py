"""Fail-closed provenance helpers for FreqDuet analysis artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("artifact primary keys must be finite")
    return value


def primary_key_sha256(
    frame: pd.DataFrame,
    primary_key: Sequence[str],
) -> str:
    columns = list(primary_key)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"artifact is missing primary-key columns: {missing}")
    if frame.duplicated(columns).any():
        raise ValueError(f"artifact primary key is not unique: {columns}")
    ordered = frame.loc[:, columns].sort_values(columns, kind="stable")
    records = [
        {column: _json_scalar(value) for column, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]
    return canonical_json_sha256(records)


def csv_artifact_record(
    path: Path | str,
    frame: pd.DataFrame,
    primary_key: Sequence[str],
) -> dict[str, object]:
    path = Path(path)
    return {
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "n_rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "primary_key": [str(column) for column in primary_key],
        "primary_key_sha256": primary_key_sha256(frame, primary_key),
    }


def validate_csv_artifact(
    path: Path | str,
    record: dict[str, object] | None,
    *,
    expected_primary_key: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing locked CSV artifact {path}")
    if not isinstance(record, dict):
        raise ValueError(f"missing artifact provenance for {path.name}")
    expected_sha = str(record.get("sha256", ""))
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise ValueError(f"{path}: SHA256 does not match its manifest")
    if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ValueError(f"{path}: byte size does not match its manifest")
    frame = pd.read_csv(path)
    if int(record.get("n_rows", -1)) != len(frame):
        raise ValueError(f"{path}: row count does not match its manifest")
    expected_columns = [str(value) for value in record.get("columns", [])]
    if expected_columns != [str(column) for column in frame.columns]:
        raise ValueError(f"{path}: columns do not match its manifest")
    manifest_key = [str(value) for value in record.get("primary_key", [])]
    if expected_primary_key is not None:
        required_key = [str(value) for value in expected_primary_key]
        if manifest_key != required_key:
            raise ValueError(f"{path}: primary key does not match the protocol")
    if not manifest_key:
        raise ValueError(f"{path}: manifest has no primary key")
    observed_key_sha = primary_key_sha256(frame, manifest_key)
    if observed_key_sha != str(record.get("primary_key_sha256", "")):
        raise ValueError(f"{path}: primary-key set does not match its manifest")
    return frame


def runtime_environment() -> dict[str, object]:
    packages = {}
    for package in ("numpy", "pandas", "PyYAML", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }


def files_fingerprint(
    root: Path | str,
    relative_paths: Iterable[str],
) -> dict[str, object]:
    root = Path(root).resolve()
    entries = []
    for relative in sorted(set(str(value) for value in relative_paths)):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"analysis fingerprint input missing: {path}")
        entries.append({
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        })
    return {
        "sha256": canonical_json_sha256(entries),
        "file_count": len(entries),
        "files": entries,
    }
