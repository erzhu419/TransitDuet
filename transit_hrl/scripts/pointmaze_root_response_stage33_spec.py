"""Frozen, independent optimizer-root qualification of the linear response."""

from argparse import Namespace

from scripts import pointmaze_budgeted_trigger_stage9_spec as original
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold


EXPERIMENT_PROTOCOL = "pointmaze_root_response_stage33_v1_qualification"
POLICY = original.POLICY
PREFLIGHT_ROOTS = (310001,)
OPTIMIZER_ROOTS = (310011, 310023, 310037, 310049, 310061, 310073, 310089, 310101)
RUNNER_SCRIPT = "scripts/run_pointmaze_root_response_stage33.py"
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = (33, 33039)
DECISION_CONTROLS = (*response.METHODS[1:], "always_keep", "always_renew")
PREDICTION_CONTROLS = (*response.METHODS[1:], "zero_value")
ENDPOINTS = tuple(f"ise:{c}" for c in DECISION_CONTROLS) + tuple(
    f"mse:{c}" for c in PREDICTION_CONTROLS)


def roots(*, preflight):
    return PREFLIGHT_ROOTS if preflight else OPTIMIZER_ROOTS


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-33 root")
    base = 4_390_000 if preflight else 4_400_000 + roster.index(root) * 10_000
    counts = (1, 1, 2, 2) if preflight else (8, 8, 8, 16)
    roles = {name: [base + offset for offset in offsets[:count]] for name, offsets, count in zip(
        ("train", "selection", "branch_fit", "trigger_eval"),
        (original._TRAIN_OFFSETS, original._SELECTION_OFFSETS,
         original._BRANCH_FIT_OFFSETS, original._TRIGGER_EVAL_OFFSETS), counts)}
    for name, offset, count in (("motion_fit", 1000, 2 if preflight else 16),
                                ("motion_eval", 1100, 2 if preflight else 8),
                                ("response_fit", 2000, 2 if preflight else 16),
                                ("response_eval", 2100, 2 if preflight else 8)):
        roles[name] = list(range(base + offset + 1, base + offset + count + 1))
    return roles


def options(root, *, preflight):
    template = original.cell_options(208001 if preflight else 209011, preflight=preflight)
    template.pop("methods")
    template.update(seed_roles(root, preflight=preflight))
    return template


def arguments(root, *, preflight):
    values = options(root, preflight=preflight)
    for role in ("train", "selection", "branch_fit", "trigger_eval"):
        values[role + "_seeds"] = values.pop(role)
    return Namespace(**values, optimizer_seed=root, preflight=preflight,
                     workers=1 if preflight else 16)


def response_cases(root, *, preflight):
    args = arguments(root, preflight=preflight)
    roles = seed_roles(root, preflight=preflight)
    generated = hold.cases_for_paths(root, roles["response_fit"], horizon=args.horizon,
                                    pairs_per_path=2 if preflight else 20)
    fit = [case for case in generated if case["check_step"] >= 64]
    query = response.query_cases(root, roles["response_eval"], horizon=args.horizon,
                                pairs_per_path=1 if preflight else 15)
    return fit, query, len(generated) - len(fit)


def budget(root, *, preflight):
    args = arguments(root, preflight=preflight)
    roles = seed_roles(root, preflight=preflight)
    checkpoints = sum((i + 1) % args.checkpoint_evaluation_interval == 0 or i == args.iterations - 1
                      for i in range(args.iterations))
    fit, query, excluded = response_cases(root, preflight=preflight)
    training = args.iterations * len(roles["train"]) * args.horizon
    selection = (1 + checkpoints) * len(roles["selection"]) * args.horizon
    diagnostics = 3 * (len(roles["branch_fit"]) + len(roles["trigger_eval"])) * args.horizon
    pairs = {name: sum(2 * (c["check_step"] + hold.SETTLEMENT_STEPS) for c in cases)
             for name, cases in (("fit", fit), ("query", query))}
    return {"training_primitive_steps": training, "selection_primitive_steps": selection,
            "diagnostic_primitive_steps": diagnostics,
            "controller_total_primitive_steps": training + selection + diagnostics,
            "factual_replay_primitive_steps": args.horizon,
            "fit_pair_primitive_steps": pairs["fit"], "query_pair_primitive_steps": pairs["query"],
            "qualification_total_primitive_steps": args.horizon + sum(pairs.values()),
            "fit_pairs": len(fit), "query_pairs": len(query),
            "excluded_incomplete_fit_prefixes": excluded,
            "candidate_proposal_inference_calls": len(fit) + len(query),
            "motion_linear_solves": 3, "response_linear_solves": 7,
            "scalar_rhs_count": 3 * 5 * 6 + 7 * 5}


def qualification_contract():
    return {"methods": list(response.METHODS), "kernel_correction": False,
            "statistical_unit": "optimizer_seed_root", "root_weighting": "equal",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": list(BOOTSTRAP_SEED), "familywise_alpha": .05,
            "interval": "two_sided_percentile_bonferroni_15_endpoints",
            "gate": "all_15_simultaneous_lower_bounds_strictly_positive",
            "retain_all_roots_regardless_of_controller_or_motion_metrics": True,
            "sequential_extension": "forbidden", "policy_deployment": False}
