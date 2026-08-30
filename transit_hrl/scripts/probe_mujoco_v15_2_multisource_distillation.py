#!/usr/bin/env python3
"""Run the v15.2 multi-source raw-policy development protocol."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from scripts import mujoco_v15_2_multisource_distillation_preflight_spec as spec
from scripts import probe_mujoco_raw_policy_distillation as base


PROBE_VERSION = spec.PROBE_VERSION


@contextmanager
def configured_base() -> Iterator[Any]:
    previous_spec = base.spec
    previous_version = base.PROBE_VERSION
    try:
        base.spec = spec
        base.PROBE_VERSION = PROBE_VERSION
        yield base
    finally:
        base.spec = previous_spec
        base.PROBE_VERSION = previous_version


def run_probe(**kwargs):
    with configured_base() as probe:
        return probe.run_probe(**kwargs)


def main() -> None:
    with configured_base() as probe:
        probe.main()


if __name__ == "__main__":
    main()
