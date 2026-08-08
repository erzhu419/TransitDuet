#!/usr/bin/env python3
"""Submit or merge source-bound Freq-HRL v7.4 confirmatory cells."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from freq_hrl.experiments.trading import full_method_confirmatory_plan_v74 as plan
from scripts import full_method_confirmatory_v74_compat as confirm
from scripts import submit_full_method_confirmatory_v732_scheduleurm as _source


def _load_private_launcher():
    name = "scripts._submit_full_method_confirmatory_v74_engine"
    spec = importlib.util.spec_from_file_location(name, Path(_source.__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen confirmatory launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_private_launcher()
launcher.plan = plan
launcher.confirm = confirm
launcher.CONFIRMATORY_MODULE = (
    "scripts.full_method_confirmatory_v74_compat"
)
launcher.SIGNATURE_VERSION = "confirmatory-v7-4-source-bound"
launcher.SUBMIT_SCRIPT_PATH = Path(__file__).resolve()


def _runtime_frozen_path(args, variant_id: str, replicate: int) -> str:
    return (
        f"/tmp/freq_hrl_v74_{args.frozen_config_sha256[:16]}_"
        f"{variant_id}_{int(replicate)}.json"
    )


launcher._runtime_frozen_path = _runtime_frozen_path
_base_build_parser = launcher.build_parser


def build_parser():
    parser = _base_build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "project":
            action.default = "Freq-HRL-v7.4-Confirmatory"
    return parser


launcher.build_parser = build_parser
experiment_cells = launcher.experiment_cells
cell_relative_dir = launcher.cell_relative_dir
build_training_command = launcher.build_training_command
build_scheduler_spec = launcher.build_scheduler_spec
expected_cell_dirs = launcher.expected_cell_dirs
merge_results = launcher.merge_results
normalize_args = launcher.normalize_args


def main() -> None:
    launcher.main()


if __name__ == "__main__":
    main()
