"""Frozen two-root continuation-value screen."""

from freq_hrl.experiments.pointmaze_continuation_credit import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_continuation_credit.py"
CONTINUATION = "frozen_stage12; path_crossfitted_value; short_only_control; fixed_alpha100_zero_threshold"
EVIDENCE_ROLE = "crossfitted_continuation_development_only"


def sampling_options(*, preflight: bool) -> dict[str, int]:
    return {"pairs_per_seed": 2 if preflight else 12}
