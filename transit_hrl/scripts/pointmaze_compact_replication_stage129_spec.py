"""Frozen compact native upper method on six additional source-policy roots."""

from scripts import pointmaze_credit_transfer_stage128_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
ROOTS = (410037, 410049, 410061, 410073, 410089, 410101)
PROTOCOL = "pointmaze_compact_replication_stage129_v1"
PANELS, METHODS, VARIANTS, CONTRASTS = source.PANELS, source.METHODS, source.VARIANTS, source.CONTRASTS
WORKERS, LABEL_SCENARIOS, TRAINING_SCENARIOS, EVALUATION_EPISODES = 16, 4, 16, 64
DAMPING, FISHER_RADIUS, MINIMUM_GAIN = source.DAMPING, source.FISHER_RADIUS, source.MINIMUM_GAIN
EPSILON, ACTION_DIM = .005, 8
PRIMARY_CONTRASTS = (("compact", "source_forecast"), ("compact", "raw"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in PRIMARY_CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, 129129


def arguments(root):
    return source.arguments(root)


def label_roles(root):
    base = 129100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1,
        "noise_seeds": {"A": base + 10001 + i, "B": base + 20001 + i}} for i in range(LABEL_SCENARIOS)]


def queries(root, period):
    return [{"scenario_seed": r["scenario_seed"], "noise_seed": r["noise_seeds"][panel], "panel": panel, "start": start}
        for r in label_roles(root) for panel in PANELS for start in range(0, arguments(root).horizon, period)]


def training_roles(root):
    base = 129100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + 30001 + i,
        "noise_seeds": {"A": base + 40001 + i, "B": base + 50001 + i}} for i in range(TRAINING_SCENARIOS)]


def evaluation_seeds(root):
    base = 129100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    h, n, e, periods = arguments(ROOTS[0]).horizon, TRAINING_SCENARIOS, EVALUATION_EPISODES, len(PERIODS)
    counts = [len(queries(ROOTS[0], p)) for p in PERIODS]
    label = sum(counts) * (1 + 2 * ACTION_DIM)
    training, crossfit, evaluation = periods * 2 * n * 5, periods * 2 * n * 3, periods * e * 6
    total = label + training + crossfit + evaluation
    return {"label_queries": sum(counts), "label_episodes": label, "training_pair_groups": periods * 2 * n,
        "training_episodes": training, "crossfit_pair_groups": periods * 2 * n, "crossfit_episodes": crossfit,
        "evaluation_pair_groups": periods * e, "evaluation_episodes": evaluation, "native_episodes": total,
        "native_steps": total * h, "native_donor_response_calls": 2 * total * h,
        "native_upper_calls": sum((q * (1 + 2 * ACTION_DIM) + 2 * n * 4 + 2 * n * 2 + e * 3) * (h // p)
            for p, q in zip(PERIODS, counts)), "empirical_fisher_solves": periods * 2,
        "fisher_jvp_batches": periods * 2, "exact_kl_forward_batches": periods * 4,
        "native_return_fits": periods * 6, "upper_candidate_weight_steps": periods * 11,
        "checkpoint_writes": periods * 2, "native_trace_writes": 0, "optimizer_steps": 0}
