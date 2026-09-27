"""Same revealed roots and sampled bins as Stage-13; adaptive continuation."""

from freq_hrl.experiments.pointmaze_adaptive_pair_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_adaptive_pair_diagnostic.py"
CONTINUATION = "frozen_stage12_trigger_on_each_arms_own_observations"
