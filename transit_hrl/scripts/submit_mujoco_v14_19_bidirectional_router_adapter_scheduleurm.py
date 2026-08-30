#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.19 bidirectional router adapter screen."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import mujoco_v14_19_bidirectional_router_adapter_screen_spec as spec
from scripts.analyze_mujoco_v14_19_bidirectional_router_adapter_screen import (
    analyze_run,
)
from scripts import submit_mujoco_v14_18_router_probe_scheduleurm as base


LAUNCHER_PATH = Path(__file__).resolve()
SIGNATURE_VERSION = "mujoco-v14-19-bidirectional-router-adapter-screen-v1"


def _overrides() -> dict[str, Any]:
    return {
        "spec": spec,
        "analyze_run": analyze_run,
        "SIGNATURE_VERSION": SIGNATURE_VERSION,
        "LAUNCHER_PATH": LAUNCHER_PATH,
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


def selected_cells():
    with configured_base() as launcher:
        return launcher.selected_cells()


def build_probe_command(args, environment, seed):
    with configured_base() as launcher:
        return launcher.build_probe_command(args, environment, seed)


def build_scheduler_spec(args, environment, seed):
    with configured_base() as launcher:
        return launcher.build_scheduler_spec(args, environment, seed)


def main() -> None:
    for name, value in _overrides().items():
        setattr(base, name, value)
    base.main()


if __name__ == "__main__":
    main()
