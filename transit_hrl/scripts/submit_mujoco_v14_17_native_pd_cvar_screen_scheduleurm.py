#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.17 mechanism screen through scheduleurm."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as spec
from scripts import (
    submit_mujoco_v14_15_closed_loop_restoration_filter_screen_scheduleurm
    as base,
)


LAUNCHER_PATH = Path(__file__).resolve()
SPEC_PATH = Path(spec.__file__).resolve()
ANALYZER_PATH = LAUNCHER_PATH.with_name(
    "analyze_mujoco_v14_17_native_pd_cvar_screen.py"
)
SIGNATURE_VERSION = "mujoco-v14-17-native-pd-cvar-screen-v1"


def _overrides() -> dict[str, Any]:
    return {
        "spec": spec,
        "multiseed_spec": spec,
        "SPEC_PATH": SPEC_PATH,
        "MULTISEED_SPEC_PATH": SPEC_PATH,
        "MULTISEED_ANALYZER_PATH": ANALYZER_PATH,
        "LAUNCHER_PATH": LAUNCHER_PATH,
        "SIGNATURE_VERSION": SIGNATURE_VERSION,
    }


@contextmanager
def configured_base() -> Iterator[Any]:
    """Install the v14.17 profile without contaminating launcher tests."""

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


def frozen_execution_identity(args):
    with configured_base() as launcher:
        return launcher.frozen_execution_identity(args)


def selected_experiment_cells(args):
    with configured_base() as launcher:
        return launcher.selected_experiment_cells(args)


def main() -> None:
    for name, value in _overrides().items():
        setattr(base, name, value)
    base.main()


if __name__ == "__main__":
    main()
