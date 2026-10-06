"""Matched native mean geometry and full-policy step calibration."""

from scripts import pointmaze_policy_curvature_stage125_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_native_geometry_stage126_v1"
WORKERS, TRAINING_SCENARIOS, EVALUATION_EPISODES = 4, 16, 32
PANELS, METHODS = ("A", "B"), ("euclidean", "natural")
DAMPING = 1.
MINIMUM_GAIN = source.MINIMUM_GAIN
FISHER_RADIUS = source.source.FISHER_RADIUS
VARIANTS = ("source_flat", "source_forecast", "euclidean", "natural", "natural_descent", "natural_blinded")
CONTRASTS = (("natural", "source_flat"), ("natural", "source_forecast"), ("natural", "euclidean"),
    ("natural", "natural_descent"), ("natural", "natural_blinded"))


def arguments(root):
    return source.arguments(root)


def label_result(root):
    return source.source.source_result(root)


def training_roles(root):
    base = 126100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1,
        "noise_seeds": {"A": base + 20001 + i, "B": base + 30001 + i}}
        for i in range(TRAINING_SCENARIOS)]


def evaluation_seeds(root):
    base = 126100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    h, n, e, periods = arguments(ROOTS[0]).horizon, TRAINING_SCENARIOS, EVALUATION_EPISODES, len(PERIODS)
    queries = len(source.source.source.queries(ROOTS[0]))
    replay, training, crossfit, evaluation = periods * queries, periods * 2 * n * 5, periods * 2 * n * 3, periods * e * 6
    return {"label_cache_loads": 1, "upper_checkpoint_loads": 2 * periods, "training_state_replays": replay,
        "training_pair_groups": periods * 2 * n, "training_episodes": training,
        "crossfit_pair_groups": periods * 2 * n, "crossfit_episodes": crossfit,
        "evaluation_pair_groups": periods * e, "evaluation_episodes": evaluation,
        "native_episodes": replay + training + crossfit + evaluation,
        "native_steps": (replay + training + crossfit + evaluation) * h,
        "native_donor_response_calls": 2 * (replay + training + crossfit + evaluation) * h,
        "native_upper_calls": sum((queries + 2 * n * 4 + 2 * n * 2 + 3 * e) * (h // p) for p in PERIODS),
        "empirical_fisher_solves": periods, "fisher_jvp_batches": periods, "exact_kl_forward_batches": 2 * periods,
        "training_direction_forward_batches": 2 * periods, "native_return_fits": 6 * periods,
        "upper_candidate_weight_steps": 9 * periods, "checkpoint_writes": periods,
        "native_trace_writes": 0, "optimizer_steps": 0}
