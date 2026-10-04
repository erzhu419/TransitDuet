"""Current-lower response to the inherited residual-plan scale; no learning."""
from scripts import pointmaze_crossed_advice_stage108_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, task_options = source.roots, source.arguments, source.task_options
EXPERIMENT_PROTOCOL = "pointmaze_control_response_stage109_v1"
POLICY = "frozen_current_lower_common_forecast_states"
RUNNER_SCRIPT = "scripts/run_pointmaze_control_response_stage109.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_control_response_stage109.py"
METHODS = {}
STRIDE, EPSILON, KL_REFERENCE = 5, .25, .001
SCALES = ("legacy", "full")
PROBES = ("forecast", "blind", "zero", *(f"{s}_{m}" for s in SCALES for m in
    ("mean", "sample", "noise", *(f"axis{i}_{sign}" for i in range(4) for sign in ("plus", "minus")))))


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "evaluation_episodes": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 109000000 if preflight else 109100000 + roots(preflight=False).index(root)*100000
    return {"native_evaluation": list(range(base+95001, base+95001+options(preflight=preflight)["evaluation_episodes"])),
        "probe_noise": "NumPy_SeedSequence_109_root_scenario_period_same_z_for_mean_plus_std_z_and_std_z"}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = len(PERIODS)*o["evaluation_episodes"]
    renewals = o["evaluation_episodes"]*sum(h//p for p in PERIODS)
    return {"native_episodes": n, "native_steps": n*h, "native_lower_calls": n*h, "native_upper_calls": 0,
        "probe_states": n*len(range(0, h, STRIDE)), "probe_lower_mean_rows": n*len(range(0, h, STRIDE))*len(PROBES),
        "probe_upper_distribution_rows": renewals, "probe_forecast_reconstructions": renewals,
        "probe_residual_curve_decodes": (len(PROBES)-2)*renewals,
        "forecast_ols_fits": o["evaluation_episodes"]*sum(h//p-1 for p in PERIODS),
        "forecast_ridge_predictions": o["evaluation_episodes"]*sum(h//p-1 for p in PERIODS),
        "worker_bounds_environment_constructions": o["workers"], "native_network_checks": n, "probe_network_checks": n,
        "counterfactual_feedback_checks": n, "forecast_replay_checks": n, "zero_action_identity_checks": n,
        "source_freeze_checks": 2}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "donors": "all_Stage107_final_learned_hint_lowers_Stage106_upper_no_selection",
        "task": source.contract()["task"], "periods": list(PERIODS),
        "trajectory": "frozen_learned_hint_lower_executing_causal_forecast_only_H1200_prefH300",
        "states": "every5th_native_state_all392_flat_feedback_preserved_counterfactual_advice_only",
        "probe_actions": "upper_mean_mean_plus_std_z_zero_mean_std_z_and_mean_plus_minus_0.25_in_each_of4_latent_coordinates",
        "decoder": "exact_existing_Bernstein_tanh_clip_then_blend_legacy_alpha_or1_zero_action_identity",
        "probes": list(PROBES), "lower": "fixed_current_lower_raw_Gaussian_mean_and_tanh_mean_same_std",
        "metrics": "command_RMS_same_covariance_KL_q99_two_singular_values_of_central_secant_Jacobian",
        "KL_reference": KL_REFERENCE, "KL_reference_role": "existing_lower_update_scale_descriptive_only_not_a_safety_or_performance_bound",
        "statistics": "equal_root_descriptive_summary_no_reward_CI_no_pooling_no_scale_selection",
        "admission": "preflight_mechanics_only_full_all8_roots_already_frozen",
        "artifacts": "compact_JSON_no_trace_checkpoint_gradient_update_or_critic_fit",
        "limits": "same_state_policy_response_not_closed_loop_gain_reward_advantage_or_proof_of_root_cause"}
