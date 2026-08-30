#!/usr/bin/env python3
"""Submit the frozen v14.29 fresh-anchor bank through scheduleurm."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_29_fresh_anchor_spec as spec
from scripts import (
    submit_mujoco_v14_15_closed_loop_restoration_filter_screen_scheduleurm
    as base,
)

LAUNCHER_PATH = Path(__file__).resolve()
SPEC_PATH = Path(spec.__file__).resolve()
ANALYZER_PATH = LAUNCHER_PATH.with_name(
    "analyze_mujoco_v14_29_fresh_anchors.py"
)
SIGNATURE_VERSION = "mujoco-v14-29-fresh-anchor-bank-v1"
BASE_NORMALIZE_ARGS = base.normalize_args
BASE_BUILD_SCHEDULER_SPEC = base.build_scheduler_spec


def _normalize_args(args):
    normalized = BASE_NORMALIZE_ARGS(args)
    if normalized.phases != ["anchor"]:
        raise SystemExit("v14.29 fresh-anchor launcher accepts only --phases anchor")
    return normalized


def _build_scheduler_spec(args, **kwargs):
    scheduler = BASE_BUILD_SCHEDULER_SPEC(args, **kwargs)
    scheduler["stage_input_paths"] = list(dict.fromkeys([
        *scheduler["stage_input_paths"],
        str((ROOT / "scripts").resolve()),
        str((ROOT / "freq_hrl").resolve()),
    ]))
    return scheduler


def _overrides() -> dict[str, Any]:
    return {
        "spec": spec,
        "multiseed_spec": spec,
        "SPEC_PATH": SPEC_PATH,
        "MULTISEED_SPEC_PATH": SPEC_PATH,
        "MULTISEED_ANALYZER_PATH": ANALYZER_PATH,
        "LAUNCHER_PATH": LAUNCHER_PATH,
        "SIGNATURE_VERSION": SIGNATURE_VERSION,
        "normalize_args": _normalize_args,
        "build_scheduler_spec": _build_scheduler_spec,
    }


@contextmanager
def configured_base() -> Iterator[Any]:
    overrides = _overrides()
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield base
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def build_parser():
    with configured_base() as launcher:
        return launcher.build_parser()


def normalize_args(args):
    with configured_base() as launcher:
        return launcher.normalize_args(args)


def build_training_command(args, **kwargs):
    with configured_base() as launcher:
        return launcher.build_training_command(args, **kwargs)


def build_scheduler_spec(args, **kwargs):
    with configured_base() as launcher:
        return launcher.build_scheduler_spec(args, **kwargs)


def selected_experiment_cells(args):
    with configured_base() as launcher:
        return launcher.selected_experiment_cells(args)


def main() -> None:
    for name, value in _overrides().items():
        setattr(base, name, value)
    base.main()


if __name__ == "__main__":
    main()
