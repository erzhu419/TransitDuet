"""One learned upper step from cached native credit; fresh closed-loop tests."""

import math
from scripts import pointmaze_reference_counterfactual_stage123_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_native_upper_step_stage124_v1"
SOURCE_RUN = "pointmaze_reference_counterfactual_stage123_probe_20261006_r1"
WORKERS, EVALUATION_EPISODES, CHUNK_SIZE = 4, 32, 1024
FISHER_RADIUS = source.EPSILON ** 2 / (2 * source.source.UPPER_STD ** 2)
MINIMUM_GAIN = source.source.MINIMUM_GAIN
VARIANTS = ("source_flat", "source_forecast", "native_ascent", "native_descent", "native_blinded")
CONTRASTS = (("native_ascent", "source_flat"), ("native_ascent", "source_forecast"),
    ("native_ascent", "native_blinded"), ("native_ascent", "native_descent"))


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def evaluation_seeds(root):
    base = 124100000 + ROOTS.index(root) * 100000
    return list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))


def budget():
    n = len(source.queries(ROOTS[0]))
    h = source.arguments(ROOTS[0]).horizon
    replay = len(PERIODS) * n
    evaluation = len(PERIODS) * len(VARIANTS) * EVALUATION_EPISODES
    return {"label_cache_loads": 1, "training_state_replays": replay,
        "evaluation_pair_groups": len(PERIODS) * EVALUATION_EPISODES,
        "evaluation_episodes": evaluation, "native_episodes": replay + evaluation,
        "native_steps": (replay + evaluation) * h,
        "native_donor_response_calls": 2 * (replay + evaluation) * h,
        "native_upper_calls": sum((n + 2 * EVALUATION_EPISODES) * (h // p) for p in PERIODS),
        "actor_pullback_forward_batches": 2 * len(PERIODS), "actor_pullback_backward_batches": 2 * len(PERIODS),
        "fisher_jvp_batches": len(PERIODS) * math.ceil(n / CHUNK_SIZE),
        "exact_kl_forward_batches": 2 * len(PERIODS) * math.ceil(n / CHUNK_SIZE),
        "upper_candidate_weight_steps": 2 * len(PERIODS),
        "optimizer_steps": 0, "checkpoint_writes": 2 * len(PERIODS), "native_trace_writes": 0}
