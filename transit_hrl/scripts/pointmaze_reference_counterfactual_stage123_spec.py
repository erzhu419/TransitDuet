"""Small paired-native upper-credit diagnosis on the exact Stage121 channel."""

from scripts import pointmaze_joint_reference_stage121_spec as source

ROOT = source.ROOT
PROTOCOL = "pointmaze_reference_counterfactual_stage123_v1"
ROOTS = tuple(source.roots(preflight=False)[:2])
PERIODS = source.PERIODS
PANELS = ("A", "B")
STARTS = (0, 300, 600, 900)
SCENARIOS_PER_START = 2
ACTION_DIM = source.UPPER_ACTION_DIM
EPSILON = .005
WORKERS = 4
VARIANTS = ("zero", *(f"axis{i}_{sign}" for i in range(ACTION_DIM) for sign in ("plus", "minus")))


def arguments(root):
    return source.arguments(root, preflight=False)


def queries(root):
    base = 123100000 + ROOTS.index(root) * 100000
    return [{"scenario_seed": base + i + 1, "start": start,
        "prefix_noise_seed": base + 20001 + i,
        "suffix_noise_seeds": {"A": base + 30001 + 2 * i, "B": base + 30002 + 2 * i}}
        for i, start in enumerate(s for s in STARTS for _ in range(SCENARIOS_PER_START))]


def budget():
    episodes_per_query = len(PANELS) * len(VARIANTS) + 4
    n = len(PERIODS) * len(STARTS) * SCENARIOS_PER_START
    h = arguments(ROOTS[0]).horizon
    return {"queries": n, "native_episodes": n * episodes_per_query,
        "native_steps": n * episodes_per_query * h,
        "native_donor_response_calls": 2 * n * episodes_per_query * h,
        "native_upper_calls": sum(len(STARTS) * SCENARIOS_PER_START * episodes_per_query * (h // p) for p in PERIODS),
        "actor_pullback_forward_batches": 2 * len(PERIODS),
        "actor_pullback_backward_batches": 2 * len(PERIODS),
        "optimizer_steps": 0, "checkpoint_writes": 0, "native_trace_writes": 0}
