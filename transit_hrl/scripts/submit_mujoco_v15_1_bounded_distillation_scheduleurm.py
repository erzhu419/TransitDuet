#!/usr/bin/env python3
"""Submit the frozen v15.1 saturation-bounded development preflight."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterator

from scripts import mujoco_v15_1_bounded_distillation_preflight_spec as spec
from scripts.analyze_mujoco_v15_1_bounded_distillation import analyze_run
from scripts import submit_mujoco_v15_raw_policy_distillation_scheduleurm as base


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
SIGNATURE_VERSION = "mujoco-v15-1-bounded-distillation-preflight-v1"


def _build_probe_command(args, environment: str, seed: int) -> str:
    anchor = base.base.anchor_relative_dir(
        args.anchor_run_name, environment, seed
    )
    output = base.base.cell_relative_dir(
        args.run_name, environment, seed
    ) / "probe.json"
    command = [
        str(args.python_executable),
        "scripts/probe_mujoco_v15_1_bounded_distillation.py",
        "--checkpoint", str(anchor / "checkpoint.pt"),
        "--summary", str(anchor / "cell_summary.json"),
        "--output", str(output),
        "--distill-roots", ",".join(map(str, spec.DISTILL_ROOTS)),
        "--design-roots", ",".join(map(str, spec.DESIGN_ROOTS)),
        "--validation-roots", ",".join(map(str, spec.VALIDATION_ROOTS)),
        "--design-fold-count", str(spec.DESIGN_FOLD_COUNT),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--leakage-cost-mode", spec.LEAKAGE_COST_MODE,
        "--risk-mode", spec.RISK_MODE,
        "--cvar-alpha", str(spec.CVAR_ALPHA),
        "--minimum-merit-reduction", str(spec.MINIMUM_MERIT_REDUCTION),
        "--funnel-multiplier", str(spec.FUNNEL_MULTIPLIER),
        "--workers", str(spec.WORKERS),
    ]
    environment_variables = [
        "MUJOCO_GL=egl", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=",
    ]
    return " ".join([*environment_variables, shlex.join(command)]) + " && echo DONE"


def _write_preregistration(args) -> None:
    target = ROOT / "results" / args.run_name
    target.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    payload = {
        "status": "frozen_before_v15_1_development_outcome_access",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "launcher_source_revision": revision,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "failed_predecessor_run": (
            "mujoco_v15_raw_policy_distillation_preflight_20260830_r1"
        ),
        "anchor_run_name": args.anchor_run_name,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "distill_roots": list(spec.DISTILL_ROOTS),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "candidates": list(spec.CANDIDATES),
        "selection_contract": spec.SELECTION_CONTRACT,
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "allowed_nodes": list(args.nodes),
            "require_node": None,
            "cpu_per_task": spec.CPU_PER_TASK,
            "ram_mb_per_task": spec.RAM_MB_PER_TASK,
            "slurm_used": False,
        },
    }
    (target / "preregistration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _module_overrides() -> dict[str, Any]:
    return {
        "spec": spec,
        "analyze_run": analyze_run,
        "_build_probe_command": _build_probe_command,
        "_write_preregistration": _write_preregistration,
        "SIGNATURE_VERSION": SIGNATURE_VERSION,
        "LAUNCHER_PATH": LAUNCHER_PATH,
    }


@contextmanager
def configured_base() -> Iterator[Any]:
    overrides = _module_overrides()
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
    return _build_probe_command(args, environment, seed)


def build_scheduler_spec(args, environment, seed):
    with configured_base() as launcher:
        return launcher.build_scheduler_spec(args, environment, seed)


def main() -> None:
    with configured_base() as launcher:
        launcher.main()


if __name__ == "__main__":
    main()
