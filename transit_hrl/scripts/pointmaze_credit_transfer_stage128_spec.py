"""Scene-held-out native credit and fixed causal history compression."""

from scripts import pointmaze_credit_coverage_stage127_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_credit_transfer_stage128_v1"
SOURCE_RUN = "pointmaze_credit_coverage_stage127_pilot_20261006_r1"
PANELS, METHODS = source.PANELS, ("raw", "compact")
WORKERS, TRAINING_SCENARIOS, EVALUATION_EPISODES = 16, 16, 32
DAMPING, FISHER_RADIUS, MINIMUM_GAIN, AUDIT_SCALE = source.DAMPING, source.FISHER_RADIUS, source.MINIMUM_GAIN, source.AUDIT_SCALE
VARIANTS = ("source_flat", "source_forecast", "raw", "compact", "compact_descent", "compact_blinded")
CONTRASTS = (("compact", "source_flat"), ("compact", "source_forecast"), ("compact", "raw"),
    ("compact", "compact_descent"), ("compact", "compact_blinded"))


def arguments(root):
    return source.arguments(root)


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def training_roles(root):
    base = 128100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1,
        "noise_seeds": {"A": base + 20001 + i, "B": base + 30001 + i}}
        for i in range(TRAINING_SCENARIOS)]


def evaluation_seeds(root):
    base = 128100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    h, n, e, periods = arguments(ROOTS[0]).horizon, TRAINING_SCENARIOS, EVALUATION_EPISODES, len(PERIODS)
    counts = [len(source.queries(ROOTS[0], p)) for p in PERIODS]
    folds = source.LABEL_SCENARIOS
    replay, audit = sum(counts), periods * folds * 2 * 4
    training, crossfit, evaluation = periods * 2 * n * 5, periods * 2 * n * 3, periods * e * 6
    total = replay + audit + training + crossfit + evaluation
    return {"source_cell_loads": 1, "training_state_replays": replay,
        "leave_scene_out_fits": periods * folds * 2, "transfer_audit_pair_groups": audit // 4,
        "transfer_audit_episodes": audit, "training_pair_groups": periods * 2 * n, "training_episodes": training,
        "crossfit_pair_groups": periods * 2 * n, "crossfit_episodes": crossfit,
        "evaluation_pair_groups": periods * e, "evaluation_episodes": evaluation,
        "native_episodes": total, "native_steps": total * h, "native_donor_response_calls": 2 * total * h,
        "native_upper_calls": sum((q + folds * 2 * 4 + 2 * n * 4 + 2 * n * 2 + e * 3) * (h // p)
            for p, q in zip(PERIODS, counts)), "empirical_fisher_solves": periods * (folds + 1) * 2,
        "fisher_jvp_batches": periods * (folds + 1) * 2, "exact_kl_forward_batches": periods * (folds + 1) * 4,
        "native_return_fits": periods * 6, "upper_candidate_weight_steps": periods * (8 * folds + 11),
        "checkpoint_writes": periods * 2, "native_trace_writes": 0, "optimizer_steps": 0}
