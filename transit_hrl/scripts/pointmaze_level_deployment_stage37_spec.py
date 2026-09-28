"""Frozen upper/lower isolation and cached gate-deployment diagnosis."""

import numpy as np
from scripts import pointmaze_update_isolation_stage36_spec as previous

ROOT, source, POLICY = previous.ROOT, previous.source, previous.POLICY
EXPERIMENT_PROTOCOL = "pointmaze_level_deployment_stage37_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_level_deployment_stage37.py"
OPTIMIZER_ROOTS, PREFLIGHT_ROOTS = previous.OPTIMIZER_ROOTS, previous.PREFLIGHT_ROOTS
METHODS = ("frozen", "upper_only", "lower_only", "upper_lower", "fixed50")
COMPONENTS = {"frozen": (), "upper_only": ("upper",), "lower_only": ("lower",),
              "upper_lower": ("upper", "lower"), "fixed50": ("upper", "lower")}
COHORTS = previous.COHORTS
GATE_TASK = "gate_deployment"
GATE_SOURCES = ("frozen", "gate_only", "joint")
GATE_MODES = ("threshold", "sampled")
ENDPOINTS = ("upper_only:frozen:return", "lower_only:frozen:return",
             "upper_lower:frozen:return", "interaction:return")
GATE_ENDPOINTS = tuple(f"{m}:sampled_threshold:return" for m in GATE_SOURCES) + (
    "gate_only:frozen:sampled:return", "joint:frozen:sampled:return")
ALL_ENDPOINTS = (*ENDPOINTS, *GATE_ENDPOINTS)
CI_FAMILY_SIZE = len(ALL_ENDPOINTS)
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = (37, 37039)
SHUFFLE_SEED_NAMESPACE = 37
AGGREGATE_STATUS = "stage37_level_diagnosis_complete"


def roots(*, preflight):
    return previous.roots(preflight=preflight)


def native_method(method):
    if method not in METHODS:
        raise ValueError("unregistered Stage-37 method")
    return "fixed50" if method == "fixed50" else "learned_history"


def options(*, preflight):
    return previous.options(preflight=preflight)


def gate_options(*, preflight):
    return {"paths": 2 if preflight else 32, "sample_streams": 2 if preflight else 4,
            "workers": 1 if preflight else 8, "checkpoint": "stage36_final_no_selection",
            "controller_sampling": False, "gate_modes": list(GATE_MODES)}


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-37 root")
    base = 8_390_000 if preflight else 8_400_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    return {"training": list(range(base + 1, base + 1 + opt["iterations"] * opt["rollouts_per_iteration"])),
            "selection": list(range(base + 2001, base + 2001 + opt["selection_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"])),
            "gate_evaluation": list(range(base + 4001, base + 4001 + gate_options(preflight=preflight)["paths"]))}


def gate_seed(root, seed, stream):
    return int(np.random.SeedSequence([37, root, seed, stream]).generate_state(1)[0])


def source_result(root, *, preflight):
    return previous.source_result(root, preflight=preflight)


def gate_source_result(root, method, *, preflight):
    run = "pointmaze_update_isolation_stage36_v1_" + ("preflight" if preflight else "full") + "_20260928_r1"
    return ROOT / "results" / run / "cells" / method / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    return previous.budget(preflight=preflight)


def gate_budget(*, preflight):
    opt = gate_options(preflight=preflight)
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    episodes = len(GATE_SOURCES) * opt["paths"] * (1 + opt["sample_streams"])
    return {"evaluation_episodes": episodes, "total_primitive_steps": episodes * horizon,
            "training_primitive_steps": 0, "selection_primitive_steps": 0}


def contrasts(v):
    f, u, l, b = (v[m]["episode_return"] for m in METHODS[:4])
    return [u - f, l - f, b - f, b - u - l + f]


def gate_contrasts(v):
    return [v[m]["sampled"]["episode_return"] - v[m]["threshold"]["episode_return"] for m in GATE_SOURCES] + [
        v[m]["sampled"]["episode_return"] - v["frozen"]["sampled"]["episode_return"] for m in GATE_SOURCES[1:]]


def contract():
    return {**previous.contract(), "methods": list(METHODS),
            "updated_actor_and_critic_levels": {m: list(COMPONENTS[m]) for m in METHODS},
            "frozen_gate": "stage35_initial_gate_weights_no_updates_in_all_training_arms",
            "minibatch_shuffle_seed": "numpy_seedsequence_37_optimizer_root_iteration",
            "primary_endpoints": list(ENDPOINTS), "all_primary_endpoints": list(ALL_ENDPOINTS),
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
            "interval": "two_sided_percentile_bonferroni_9_endpoints",
            "interaction": "upper_lower_minus_upper_only_minus_lower_only_plus_frozen",
            "decision": "diagnose_upper_lower_updates_and_gate_deployment_not_algorithm_success",
            "evidence_role": "conditional_level_and_deployment_diagnosis_not_independent_confirmation"}


def gate_contract():
    return {"source_protocol": previous.EXPERIMENT_PROTOCOL, "source_methods": list(GATE_SOURCES),
            "checkpoint": "final128_full_final2_preflight_no_checkpoint_selection",
            "gate_modes": list(GATE_MODES), "threshold": .5,
            "controller_actions": "deterministic_upper_and_lower_in_both_modes",
            "sampling": "bernoulli_original_probabilities_torch_seed_gate_seed_plus_observed_step",
            "gate_seed": "numpy_seedsequence_37_root_environment_seed_stream_uint32",
            "sample_stream_pairing": "same_stream_seeds_for_all_cached_models",
            "current_cost": 1., "candidate_previews": 0, "max_plan_age_steps": 100, "check_steps": 25,
            "primary_endpoints": list(GATE_ENDPOINTS), "all_primary_endpoints": list(ALL_ENDPOINTS),
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
            "familywise_alpha": .05, "interval": "two_sided_percentile_bonferroni_9_endpoints",
            "statistical_unit": "optimizer_root_means_after_path_and_stream_averaging",
            "root_exclusion": "forbidden", "sequential_extension": "forbidden",
            "evidence_role": "fresh_path_diagnosis_on_stage36_frozen_final_weights"}
