#!/usr/bin/env python3
"""Analyze the MuJoCo v15.1 saturation-bounded development preflight."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

from scripts import mujoco_v15_1_bounded_distillation_preflight_spec as spec
from scripts import analyze_mujoco_v15_raw_policy_distillation as base


PROBE_VERSION = spec.PROBE_VERSION
cell_relative_dir = base.cell_relative_dir


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


def analyze_payloads(payloads):
    with configured_base() as analyzer:
        return analyzer.analyze_payloads(payloads)


def analyze_run(run_name, output_dir=None):
    with configured_base() as analyzer:
        return analyzer.analyze_run(run_name, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "validation_supported_count": result["validation_supported_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
