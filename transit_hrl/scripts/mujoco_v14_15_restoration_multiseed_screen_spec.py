"""Frozen candidate-fixed MuJoCo v14.15 multiseed development design."""

from __future__ import annotations

import math

from scripts import (
    mujoco_v14_15_closed_loop_restoration_filter_screen_spec as base,
)


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_15_restoration_multiseed_development_screen_v1"
)
ANALYSIS_PROFILE = "fixed_v14_15_restoration_candidate_multiseed_v1"
FROZEN_EXECUTION_REVISION = "deb798c56fbb1fce06a8c84bbac84c77d3556703"
FROZEN_EXECUTION_SOURCE_MANIFEST_SHA256 = (
    "1708dd24b9a7badc641b8b188b111020f5d4dd10e73338d591b032a4f64cfc91"
)
SELECTION_SOURCE_EVIDENCE_ID = (
    "mujoco_v14_15_closed_loop_restoration_filter_preflight"
)
SELECTION_SOURCE_DECISION_SHA256 = (
    "540036f775d4fce2c10f0d9aa854da4209cf4e5c8801ff87430713493fb16fdf"
)
PRESELECTED_CANDIDATE_ARM = (
    "group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3"
)
STRICT_ABLATION_ARM = base.STRICT_CLOSED_LOOP_CONTROL_ARM
MATCHED_COMPARATOR_ARM = base.MATCHED_COMPARATOR_ARM
CALIBRATION_ARM = base.CALIBRATION_ARM
BASE_CONTROL_ARM = base.BASE_CONTROL_ARM

ENVIRONMENTS = base.ENVIRONMENTS
# The first optimizer seed selected the candidate and is permanently excluded.
OPTIMIZER_SEEDS = base.OPTIMIZER_SEEDS[1:]
ARMS = tuple(base.ARMS)
EVALUATION_DISTURBANCE_MODES = base.EVALUATION_DISTURBANCE_MODES
FREQUENCY_METRICS = (
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
)

CONFIDENCE = 0.95
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 3_621_857_943
RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
FREQUENCY_REDUCTION_FRACTION = 0.05
FREQUENCY_LOG_REDUCTION_THRESHOLD = -math.log(
    1.0 - FREQUENCY_REDUCTION_FRACTION
)
METRIC_EPSILON = 1e-12

MINIMUM_ENVIRONMENT_COMPLETE_GATE_COUNT = 12
MINIMUM_AGGREGATE_COMPLETE_GATE_FRACTION_LOWER = 0.70
WILSON_ONE_SIDED_Z = 1.6448536269514722

PRIMARY_CONTRAST_ORDER = tuple(
    (environment, metric)
    for environment in ENVIRONMENTS
    for metric in ("normalized_episode_return", *FREQUENCY_METRICS)
)
PRIMARY_THRESHOLDS = {
    (environment, "normalized_episode_return"):
        -RETURN_NONINFERIORITY_MARGIN_FRACTION
    for environment in ENVIRONMENTS
} | {
    (environment, metric): FREQUENCY_LOG_REDUCTION_THRESHOLD
    for environment in ENVIRONMENTS
    for metric in FREQUENCY_METRICS
}

CLAIM_BOUNDARY = (
    "This candidate-fixed screen uses 15 optimizer seeds that were not used "
    "to select the v14.15 restoration arm and treats optimizer seed, not "
    "held-out path, as the statistical unit. It may authorize a new frozen "
    "confirmation protocol. It is still development evidence because the "
    "algorithm family and candidate were chosen using earlier MuJoCo outcomes."
)


def validate() -> None:
    if len(FROZEN_EXECUTION_REVISION) != 40:
        raise RuntimeError("multiseed execution revision is not frozen")
    if len(FROZEN_EXECUTION_SOURCE_MANIFEST_SHA256) != 64:
        raise RuntimeError("multiseed execution source manifest is not frozen")
    if PRESELECTED_CANDIDATE_ARM not in base.AUTHORIZING_ARMS:
        raise RuntimeError("multiseed candidate is not a v14.15 authorizing arm")
    if set(OPTIMIZER_SEEDS) & {base.OPTIMIZER_SEEDS[0]}:
        raise RuntimeError("preflight optimizer seed leaked into multiseed screen")
    if len(OPTIMIZER_SEEDS) != 15 or len(set(OPTIMIZER_SEEDS)) != 15:
        raise RuntimeError("multiseed optimizer registry is incomplete")
    if set(ARMS) != set(base.ARMS):
        raise RuntimeError("multiseed ablation registry drifted")
    if len(PRIMARY_CONTRAST_ORDER) != 18:
        raise RuntimeError("multiseed primary contrast family is incomplete")


validate()
