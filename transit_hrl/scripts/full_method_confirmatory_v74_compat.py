#!/usr/bin/env python3
"""Run the frozen v7.3.2 engine under the source-bound v7.4 plan.

The v7.4 algorithm source already contains the complete confirmatory engine and
the pre-registered v7.4 plan, but its original CLI imports the v7.3.2 plan by
name. This adapter loads a private copy of that engine, replaces only the plan
registry and versioned random-seed namespace, and records the adapter hash in
each result. It lives outside ``freq_hrl`` so the HPO-frozen algorithm manifest
is unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from freq_hrl.experiments.reproducibility import derive_seed as _derive_seed_base
from freq_hrl.experiments.trading import full_method_confirmatory_plan_v74 as plan
from freq_hrl.experiments.trading import full_method_confirmatory_v732 as _source


RUNTIME_COMPATIBILITY_VERSION = (
    "v74_plan_adapter_over_frozen_v732_engine_v1"
)
CONFIRMATORY_PROTOCOL_VERSION = (
    "full_method_confirmatory_v7_4_source_bound_training_replicate_v1"
)
CONFIRMATORY_IMPLEMENTATION_VERSION = (
    "full_method_confirmatory_robust_checkpoint_v7_4_2026_08_08"
)


def _load_private_engine():
    name = (
        "freq_hrl.experiments.trading."
        "_full_method_confirmatory_v74_frozen_engine"
    )
    spec = importlib.util.spec_from_file_location(name, Path(_source.__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen confirmatory engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _v74_derive_seed(namespace: str, *parts: object) -> int:
    normalized = str(namespace).replace("freq_hrl_v732_", "freq_hrl_v74_")
    return _derive_seed_base(normalized, *parts)


engine = _load_private_engine()
engine.plan = plan
engine.EVALUATION_SCENARIOS = tuple(plan.EVALUATION_SCENARIOS)
engine.DEFAULT_HELDOUT_SEEDS = tuple(plan.DEFAULT_HELDOUT_SEEDS)
engine.DEFAULT_CONFIRMATORY_REPLICATES = tuple(
    plan.DEFAULT_CONFIRMATORY_REPLICATES
)
engine.CONFIRMATORY_PROTOCOL_VERSION = CONFIRMATORY_PROTOCOL_VERSION
engine.CONFIRMATORY_IMPLEMENTATION_VERSION = (
    CONFIRMATORY_IMPLEMENTATION_VERSION
)
engine.derive_seed = _v74_derive_seed

run_confirmatory_cell = engine.run_confirmatory_cell
write_confirmatory_cell = engine.write_confirmatory_cell


def runtime_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _output_dir_from_argv(argv: list[str]) -> Path | None:
    try:
        index = argv.index("--output-dir")
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return Path(argv[index + 1])


def _annotate_cell(output_dir: Path | None) -> None:
    if output_dir is None:
        return
    path = Path(output_dir) / "cell_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "confirmatory_runtime_compatibility_version": (
            RUNTIME_COMPATIBILITY_VERSION
        ),
        "confirmatory_runtime_adapter_sha256": runtime_sha256(),
        "confirmatory_engine_source_module": (
            "freq_hrl.experiments.trading.full_method_confirmatory_v732"
        ),
        "confirmatory_plan_adapter_status": "v74_source_bound_before_heldout",
    })
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_confirmatory_cells(input_dirs, **kwargs):
    expected_hash = runtime_sha256()
    for directory in input_dirs:
        path = Path(directory) / "cell_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("confirmatory_runtime_compatibility_version")
            != RUNTIME_COMPATIBILITY_VERSION
            or summary.get("confirmatory_runtime_adapter_sha256")
            != expected_hash
            or summary.get("confirmatory_plan_adapter_status")
            != "v74_source_bound_before_heldout"
        ):
            raise ValueError(
                f"confirmatory runtime provenance mismatch: {directory}"
            )
    payload = engine.merge_confirmatory_cells(input_dirs, **kwargs)
    payload["summary"].update({
        "confirmatory_runtime_compatibility_version": (
            RUNTIME_COMPATIBILITY_VERSION
        ),
        "confirmatory_runtime_adapter_sha256": expected_hash,
        "confirmatory_plan_adapter_status": (
            "v74_source_bound_before_heldout"
        ),
    })
    return payload


def write_confirmatory_merge(output_dir: Path, payload) -> None:
    engine.write_confirmatory_merge(output_dir, payload)


def main() -> None:
    output_dir = _output_dir_from_argv(sys.argv[1:])
    engine.main()
    if "--merge-inputs" not in sys.argv:
        _annotate_cell(output_dir)


if __name__ == "__main__":
    main()
