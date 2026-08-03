#!/usr/bin/env python3
"""Submit Freq-HRL v5 nested-validation HPO through scheduleurm."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.trading import full_method_hpo_v5 as hpo  # noqa: E402
from scripts import submit_full_method_hpo_scheduleurm as base  # noqa: E402


def configure_base() -> None:
    base.ALL_VARIANT_IDS = hpo.ALL_VARIANT_IDS
    base.CANDIDATES_BY_ID = hpo.CANDIDATES_BY_ID
    base.DEFAULT_PILOT_SCENARIOS = hpo.DEFAULT_PILOT_SCENARIOS
    base.DEFAULT_TUNING_SEEDS = hpo.DEFAULT_TUNING_SEEDS
    base.FULL_METHOD_TUNING_PROTOCOL_VERSION = (
        hpo.FULL_METHOD_TUNING_PROTOCOL_VERSION
    )
    base.VARIANTS_BY_ID = hpo.VARIANTS_BY_ID
    base.candidate_ids_for_variant = hpo.candidate_ids_for_variant
    base.merge_hpo_cells = hpo.merge_hpo_cells
    base.write_hpo_merge = hpo.write_hpo_merge
    base.HPO_MODULE = "freq_hrl.experiments.trading.full_method_hpo_v5"
    base.HPO_SIGNATURE_VERSION = "full-hpo-v3"
    base.SMOKE_FULL_VARIANT = hpo.ABLATION_PARENT_VARIANT
    base.SUBMIT_SCRIPT_PATH = Path(__file__).resolve()


if __name__ == "__main__":
    configure_base()
    base.main()
