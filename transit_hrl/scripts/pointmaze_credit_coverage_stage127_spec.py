"""Temporal credit coverage on the unchanged bounded native reference channel."""

from scripts import pointmaze_native_geometry_stage126_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_credit_coverage_stage127_v1"
PANELS, METHODS = ("A", "B"), ("coarse", "complete")
LABEL_SCENARIOS, TRAINING_SCENARIOS, EVALUATION_EPISODES, WORKERS = 4, 16, 32, 16
COARSE_STARTS = (0, 300, 600, 900)
EPSILON, ACTION_DIM, AUDIT_SCALE = .005, 8, .1
DAMPING, FISHER_RADIUS, MINIMUM_GAIN = source.DAMPING, source.FISHER_RADIUS, source.MINIMUM_GAIN
VARIANTS = ("source_flat", "source_forecast", "coarse", "complete", "complete_descent", "complete_blinded")
CONTRASTS = (("complete", "source_flat"), ("complete", "source_forecast"), ("complete", "coarse"),
    ("complete", "complete_descent"), ("complete", "complete_blinded"))


def arguments(root):
    return source.arguments(root)


def label_roles(root):
    base = 127100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1,
        "noise_seeds": {"A": base + 10001 + i, "B": base + 20001 + i}}
        for i in range(LABEL_SCENARIOS)]


def queries(root, period):
    return [{"scenario_seed": r["scenario_seed"], "noise_seed": r["noise_seeds"][panel],
        "panel": panel, "start": start} for r in label_roles(root) for panel in PANELS
        for start in range(0, arguments(root).horizon, period)]


def training_roles(root):
    base = 127100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + 30001 + i,
        "noise_seeds": {"A": base + 40001 + i, "B": base + 50001 + i}}
        for i in range(TRAINING_SCENARIOS)]


def evaluation_seeds(root):
    base = 127100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    h, n, e, periods = arguments(ROOTS[0]).horizon, TRAINING_SCENARIOS, EVALUATION_EPISODES, len(PERIODS)
    counts = [len(queries(ROOTS[0], p)) for p in PERIODS]
    label = sum(counts) * (1 + 2 * ACTION_DIM)
    audit = periods * LABEL_SCENARIOS * len(PANELS) * 4
    training, crossfit, evaluation = periods * 2 * n * 5, periods * 2 * n * 3, periods * e * 6
    total = label + audit + training + crossfit + evaluation
    return {"label_queries": sum(counts), "label_episodes": label,
        "gradient_audit_pair_groups": audit // 4, "gradient_audit_episodes": audit,
        "training_pair_groups": periods * 2 * n, "training_episodes": training,
        "crossfit_pair_groups": periods * 2 * n, "crossfit_episodes": crossfit,
        "evaluation_pair_groups": periods * e, "evaluation_episodes": evaluation,
        "native_episodes": total, "native_steps": total * h, "native_donor_response_calls": 2 * total * h,
        "native_upper_calls": sum((q * (1 + 2 * ACTION_DIM) + LABEL_SCENARIOS * 2 * 4
            + 2 * n * 4 + 2 * n * 2 + e * 3) * (h // p) for p, q in zip(PERIODS, counts)),
        "empirical_fisher_solves": periods, "fisher_jvp_batches": periods * 2,
        "exact_kl_forward_batches": periods * 4, "native_return_fits": periods * 6,
        "upper_candidate_weight_steps": periods * 15, "checkpoint_writes": periods * 2,
        "native_trace_writes": 0, "optimizer_steps": 0}
