"""Frozen offline qualification on the two Stage-16 development roots."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_paired_value_qualification import PROTOCOL_VERSION, SOURCE_PROTOCOL
from scripts.pointmaze_continuation_credit_spec import POLICY, roots, sampling_options


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_paired_value_qualification.py"
CONTINUATION = "cached_stage16_endpoints; matched_absolute_vs_paired_loss; no_environment_sampling"
EVIDENCE_ROLE = "paired_critic_qualification_only"


def source_result(root: int, *, preflight: bool) -> Path:
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-17 root is not registered")
    phase = "preflight" if preflight else "development"
    return (Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_continuation_credit_stage16_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json")
