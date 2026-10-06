"""Train native policy-ray step size; evaluate on fresh full episodes."""

from scripts import pointmaze_native_upper_step_stage124_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_policy_curvature_stage125_v1"
SOURCE_RUN = "pointmaze_native_upper_step_stage124_pilot_20261006_r1"
WORKERS, TRAINING_SCENARIOS, EVALUATION_EPISODES = 4, 16, 32
PANELS = ("A", "B")
VARIANTS = ("source_flat", "source_forecast", "unit_ascent", "curvature_ascent",
    "curvature_descent", "curvature_blinded")
CONTRASTS = (("curvature_ascent", "source_flat"), ("curvature_ascent", "source_forecast"),
    ("curvature_ascent", "curvature_blinded"), ("curvature_ascent", "curvature_descent"),
    ("curvature_ascent", "unit_ascent"))
MINIMUM_GAIN = source.MINIMUM_GAIN


def arguments(root):
    return source.source.arguments(root)


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def training_roles(root):
    base = 125100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1,
        "noise_seeds": {"A": base + 20001 + i, "B": base + 30001 + i}}
        for i in range(TRAINING_SCENARIOS)]


def evaluation_seeds(root):
    base = 125100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    h, n, e, periods = arguments(ROOTS[0]).horizon, TRAINING_SCENARIOS, EVALUATION_EPISODES, len(PERIODS)
    training, crossfit, evaluation = periods * 2 * n * 3, periods * 2 * n * 2, periods * e * len(VARIANTS)
    return {"source_cell_loads": 1, "upper_checkpoint_loads": 2 * periods,
        "training_pair_groups": periods * 2 * n, "training_episodes": training,
        "crossfit_pair_groups": periods * 2 * n, "crossfit_episodes": crossfit,
        "evaluation_pair_groups": periods * e, "evaluation_episodes": evaluation,
        "native_episodes": training + crossfit + evaluation,
        "native_steps": (training + crossfit + evaluation) * h,
        "native_donor_response_calls": 2 * (training + crossfit + evaluation) * h,
        "native_upper_calls": sum((2 * n * 2 + 2 * n + 3 * e) * (h // p) for p in PERIODS),
        "native_return_fits": 3 * periods, "upper_candidate_weight_steps": 4 * periods,
        "checkpoint_writes": periods, "native_trace_writes": 0, "optimizer_steps": 0}
