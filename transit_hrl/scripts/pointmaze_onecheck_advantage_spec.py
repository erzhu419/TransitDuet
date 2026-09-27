"""Frozen two-root screen for adaptive one-check policy improvement."""

from freq_hrl.experiments.pointmaze_onecheck_advantage import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_onecheck_advantage.py"
CONTINUATION = "stage12_reference_policy_for_branch_labels; zero_advantage_greedy_for_deployment"
EVIDENCE_ROLE = "one_step_policy_improvement_development_only"


def sampling_options(*, preflight: bool) -> dict[str, int]:
    return {"pairs_per_seed": 2 if preflight else 12}
