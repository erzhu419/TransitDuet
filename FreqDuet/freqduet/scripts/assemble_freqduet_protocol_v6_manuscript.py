#!/usr/bin/env python3
"""Assemble the Protocol V6 paper text from the frozen evidence package.

The output is deliberately claim-bounded.  A successful build means that the
text and tables agree with the packaged V8/V9 evidence; it does not override
the failed V9 long-training gate or make the study submission-ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT / "results_freqduet" / "paper_package" / "protocol_v6_current_best"
)
DEFAULT_OUT = ROOT.parent / "paper" / "protocol_v6"
EDITORIAL_SOURCE_DIR = ROOT / "paper_sources" / "protocol_v6"

PROTOCOL = "freqduet-eval-v6"
PAPER_CONTROLLER = "F_freqduet_protocol_v6_confirmed_main_hiro"
SOURCE_CANDIDATE = "F_freqduet_protocol_v6_avlcompact_w2_hiro"
REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
TITLE = "FreqDuet: Causally Observed Frequency Allocation for Hierarchical Bus Timetable and Holding Control"
KEYWORDS = (
    "bus holding",
    "hierarchical reinforcement learning",
    "demand decomposition",
    "causal observation",
    "headway control",
    "reproducibility",
)
EDITORIAL_FILES = (
    "introduction.md",
    "related_work.md",
    "discussion.md",
    "conclusion.md",
    "availability.md",
    "terminology.md",
    "literature_verification.md",
    "journal_target.md",
    "references.bib",
    "elsarticle-template.tex",
    "supplementary-template.tex",
    "build_trc_submission.sh",
)
FIGURE_STEMS = {
    1: "method",
    2: "confirmation_robustness",
    3: "external_tradeoff",
    4: "physical_outcomes",
    5: "external_realism",
}

METRICS = (
    ("passenger_journey_min", "Restricted passenger journey", "min", 1.0, 3),
    ("passenger_wait_min", "Restricted passenger wait", "min", 1.0, 3),
    ("in_vehicle_min", "Restricted in-vehicle time", "min", 1.0, 3),
    ("headway_cv", "Headway coefficient of variation", "", 1.0, 3),
    ("unserved_rate", "Unserved passengers", "percentage points", 100.0, 2),
    ("holding_s_per_trip", "Realized holding", "s/launched trip", 1.0, 1),
    ("denied_trip_rate", "Trips denied at least once", "percentage points", 100.0, 2),
    ("restricted_service_cost", "Restricted service cost", "", 1.0, 3),
)
METRIC_SPEC = {metric: (label, unit, scale, digits) for metric, label, unit, scale, digits in METRICS}

EXPECTED_METHOD = {
    "paper_controller": PAPER_CONTROLLER,
    "protocol": PROTOCOL,
    "frequency_method": "harmonic",
    "historical_prior": True,
    "service_start_hour": 6,
    "service_end_hour": 19,
    "clearance_time_s": 14400.0,
    "fleet_size": 12,
    "bin_sec": 60.0,
    "harmonic_period_s": 50400.0,
    "fourier_k": 4,
    "harmonic_forgetting": 0.9995,
    "harmonic_prior_var": 0.01,
    "harmonic_ridge": 0.01,
    "forecast_horizon_s": 1800.0,
    "upper_frequency_authority": "low",
    "lower_frequency_authority": "high",
    "planning_horizon_s": 2700.0,
    "replan_interval_s": 900.0,
    "headway_budget_mode": "rolling_zero_sum_delta_v6",
    "terminal_dispatch": True,
    "upper_delta_min_s": -60.0,
    "upper_delta_max_s": 60.0,
    "terminal_shift_min_s": -45.0,
    "terminal_shift_max_s": 45.0,
    "uses_last_action_feature": True,
    "regularity_objective": "avl_two_sided_incremental_reward",
    "regularity_reward_weight": 2.0,
    "regularity_tolerance_fraction": 0.02,
    "regularity_cost_cap": 0.25,
    "objective_wait_metric": "restricted",
    "service_cost_weights": {
        "wait": 1.0,
        "fleet": 1.0,
        "headway": 1.0,
        "unserved": 5.0,
        "incomplete_service": 5.0,
    },
    "upper_algorithm": "pessimistic_ensemble_sac_v4",
    "upper_hidden_dim": 64,
    "upper_ensemble_size": 10,
    "upper_learning_rate": 0.0003,
    "upper_discount": 0.95,
    "upper_batch_size": 64,
    "upper_updates_per_episode": 10,
    "lower_algorithm": "pessimistic_ensemble_sac_lagrangian_v4",
    "lower_hidden_dim": 64,
    "lower_ensemble_size": 10,
    "lower_learning_rate": 0.0003,
    "lower_dual_learning_rate": 0.0001,
    "lower_discount": 0.99,
    "lower_batch_size": 512,
    "lower_updates_per_episode": 30,
    "upper_warmup_episodes": 30,
    "legacy_holding_guard_enabled": False,
    "promotion_enabled": False,
    "leakage_penalty_enabled": False,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def one_row(rows: Iterable[dict[str, str]], **matches: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if all(row.get(key) == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {matches}, found {len(selected)}")
    return selected[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bibliography_keys(bibliography: str) -> set[str]:
    return set(
        re.findall(r"^@\w+\s*\{\s*([^,\s]+)", bibliography, re.MULTILINE)
    )


def citation_keys(markdown: str) -> set[str]:
    return set(
        re.findall(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_:-]+)", markdown)
    )


def require_resolved_citations(markdown: str, bibliography: str) -> None:
    missing = sorted(citation_keys(markdown) - bibliography_keys(bibliography))
    require(not missing, f"missing bibliography keys: {', '.join(missing)}")


def read_editorial_sources(source_dir: Path = EDITORIAL_SOURCE_DIR) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name in EDITORIAL_FILES:
        path = source_dir / name
        require(path.is_file(), f"missing editorial source: {name}")
        sources[name] = path.read_text()

    expected_headings = {
        "introduction.md": "# Introduction",
        "related_work.md": "# Related Work",
        "discussion.md": "# Discussion",
        "conclusion.md": "# Conclusions",
        "availability.md": "# Data and Code Availability",
    }
    for name, heading in expected_headings.items():
        require(sources[name].startswith(heading), f"unexpected heading in {name}")

    require_resolved_citations(
        "\n".join(
            sources[name]
            for name in ("introduction.md", "related_work.md", "discussion.md")
        ),
        sources["references.bib"],
    )
    require(
        "The experiments do not establish that this interface caused" in sources["discussion.md"],
        "frequency-claim boundary is missing from Discussion",
    )
    require(
        "V9 did not confirm" in sources["discussion.md"],
        "V9 failure is missing from Discussion",
    )
    require(
        "fixed headway remained better" in sources["conclusion.md"],
        "fixed-headway trade-off is missing from Conclusions",
    )
    return sources


def validate_package(package_dir: Path) -> dict[str, Any]:
    status = read_json(package_dir / "evidence_status.json")
    require(status.get("protocol") == PROTOCOL, "paper package protocol mismatch")
    require(
        status.get("paper_controller") == PAPER_CONTROLLER,
        "paper package controller mismatch",
    )
    require(status.get("v8_confirmation_status") == "unique_pass", "V8 is not confirmed")
    require(
        status.get("v9_longtrain_status") == "longtrain_not_confirmed",
        "V9 negative result is missing",
    )
    require(status.get("submission_ready") is False, "V9 hold was unexpectedly overridden")
    require(
        status.get("submission_blocker") == "v9_longtrain_not_confirmed",
        "submission blocker mismatch",
    )

    method = read_json(
        package_dir / "figures" / "source_data" / "figure1_method_contract.json"
    )
    for key, expected in EXPECTED_METHOD.items():
        require(method.get(key) == expected, f"method contract mismatch for {key}")
    require(
        method.get("action_bins_s") == [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
        "holding action alphabet mismatch",
    )

    v8 = read_rows(package_dir / "tables" / "table1_v8_confirmation.csv")
    v9 = read_rows(package_dir / "tables" / "table2_v9_longtrain.csv")
    external = read_rows(package_dir / "tables" / "table3_v9_external_baselines.csv")
    decisions = read_rows(package_dir / "tables" / "table4_evidence_decisions.csv")
    require(len(v8) == len(METRICS), "V8 table metric roster mismatch")
    require(len(v9) == len(METRICS), "V9 table metric roster mismatch")
    require(len(external) == 3 * len(METRICS), "external table metric roster mismatch")

    for rows, pairs, phase, candidate in (
        (v8, "24", "v8_independent_confirmation_ep40", SOURCE_CANDIDATE),
        (v9, "64", "v9_independent_longtrain_ep200", PAPER_CONTROLLER),
    ):
        require({row["metric"] for row in rows} == set(METRIC_SPEC), f"{phase} metrics mismatch")
        require(all(row.get("n_pairs") == pairs for row in rows), f"{phase} pair count mismatch")
        require(all(row.get("phase") == phase for row in rows), f"{phase} label mismatch")
        require(
            all(row.get("paper_controller") == PAPER_CONTROLLER for row in rows),
            f"{phase} paper controller mismatch",
        )
        require(all(row.get("reference") == REFERENCE for row in rows), f"{phase} reference mismatch")
        require(all(row.get("source_candidate") == candidate for row in rows), f"{phase} candidate mismatch")

    for baseline in ("fixed_headway", "rule_holding", "rule_mpc"):
        subset = [row for row in external if row.get("baseline") == baseline]
        require(len(subset) == len(METRICS), f"external metric roster mismatch for {baseline}")
        require(
            {row.get("metric") for row in subset} == set(METRIC_SPEC),
            f"external metrics mismatch for {baseline}",
        )
        require(
            all(row.get("paper_controller") == PAPER_CONTROLLER for row in subset),
            f"external paper controller mismatch for {baseline}",
        )
        require(all(row.get("n_pairs") == "64" for row in subset), "external pair count mismatch")

    require(len(decisions) == 2, "decision ledger must contain exactly V8 and V9")
    v8_decision = one_row(decisions, phase="v8_independent_confirmation_ep40")
    v9_decision = one_row(decisions, phase="v9_independent_longtrain_ep200")
    require(v8_decision.get("decision") == "primary_confirmed", "V8 decision mismatch")
    require(v8_decision.get("claim_eligible") == "True", "V8 eligibility mismatch")
    require(v8_decision.get("controller") == PAPER_CONTROLLER, "V8 ledger controller mismatch")
    require(
        (v8_decision.get("train_seeds"), v8_decision.get("evaluation_seeds"),
         v8_decision.get("paired_rollouts")) == ("6", "4", "24"),
        "V8 ledger seed contract mismatch",
    )
    require(v9_decision.get("decision") == "longtrain_not_confirmed", "V9 decision mismatch")
    require(v9_decision.get("claim_eligible") == "False", "V9 eligibility mismatch")
    require(v9_decision.get("controller") == PAPER_CONTROLLER, "V9 ledger controller mismatch")
    require(
        (v9_decision.get("train_seeds"), v9_decision.get("evaluation_seeds"),
         v9_decision.get("paired_rollouts")) == ("8", "8", "64"),
        "V9 ledger seed contract mismatch",
    )

    v9_summary = one_row(
        read_rows(
            package_dir
            / "source_artifacts"
            / "v9"
            / "frozen_summary.csv"
        ),
        config=PAPER_CONTROLLER,
    )
    require(
        float(v9_summary.get("trip_launch_rate_mean", "nan")) == 1.0,
        "V9 learned-policy launch rate is not complete",
    )
    require(
        float(v9_summary.get("trip_completion_rate_mean", "nan")) == 1.0,
        "V9 learned-policy completion rate is not complete",
    )

    for stem in (
        "fig1_protocol_v6_method",
        "fig2_protocol_v6_confirmation_robustness",
        "fig3_protocol_v6_external_tradeoff",
        "fig4_protocol_v6_physical_outcomes",
        "fig5_protocol_v6_external_realism",
    ):
        require((package_dir / "figures" / f"{stem}.png").is_file(), f"missing figure: {stem}.png")
        require((package_dir / "figures" / f"{stem}.pdf").is_file(), f"missing figure: {stem}.pdf")

    return {
        "status": status,
        "method": method,
        "v8": v8,
        "v9": v9,
        "external": external,
        "decisions": decisions,
        "v9_summary": v9_summary,
    }


def signed(value: Any, digits: int) -> str:
    return f"{float(value):+.{digits}f}"


def p_value(value: Any) -> str:
    number = float(value)
    return "<0.001" if number < 0.001 else f"{number:.3f}"


def effect(row: dict[str, str], delta_key: str, metric: str) -> str:
    _, _, scale, digits = METRIC_SPEC[metric]
    delta = float(row[delta_key]) * scale
    low = float(row["ci95_low"]) * scale
    high = float(row["ci95_high"]) * scale
    return f"{signed(delta, digits)} [{signed(low, digits)}, {signed(high, digits)}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def latex_table(headers: list[str], rows: list[list[str]], alignment: str) -> str:
    rendered = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    rendered.extend(
        " & ".join(latex_escape(item) for item in row) + r" \\" for row in rows
    )
    rendered.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rendered)


def main_confirmation_table(v8: list[dict[str, str]], v9: list[dict[str, str]]) -> tuple[str, str]:
    rows: list[list[str]] = []
    for metric, label, unit, _, _ in METRICS:
        v8_row = one_row(v8, metric=metric)
        v9_row = one_row(v9, metric=metric)
        display = f"{label} ({unit})" if unit else label
        rows.append([
            display,
            effect(v8_row, "delta_candidate_minus_reference", metric),
            p_value(v8_row["paired_signflip_p_holm"]),
            effect(v9_row, "delta_candidate_minus_reference", metric),
            p_value(v9_row["paired_signflip_p_holm"]),
        ])
    headers = [
        "Outcome",
        "V8 delta [95% CI]",
        "V8 Holm p",
        "V9 delta [95% CI]",
        "V9 Holm p",
    ]
    return markdown_table(headers, rows), latex_table(headers, rows, "lrrrr")


def external_main_table(external: list[dict[str, str]]) -> tuple[str, str]:
    core_metrics = (
        "passenger_journey_min",
        "headway_cv",
        "denied_trip_rate",
        "restricted_service_cost",
    )
    rows: list[list[str]] = []
    names = {
        "fixed_headway": "Fixed headway",
        "rule_holding": "Rule holding",
        "rule_mpc": "Rule MPC",
    }
    for baseline in names:
        row = [names[baseline]]
        for metric in core_metrics:
            record = one_row(external, baseline=baseline, metric=metric)
            row.append(effect(record, "delta_learned_minus_baseline", metric))
        rows.append(row)
    headers = [
        "Baseline",
        "Journey min",
        "Headway CV",
        "Denied trips (pp)",
        "Service cost",
    ]
    return markdown_table(headers, rows), latex_table(headers, rows, "lrrrr")


def external_full_table(external: list[dict[str, str]]) -> tuple[str, str]:
    names = {
        "fixed_headway": "Fixed headway",
        "rule_holding": "Rule holding",
        "rule_mpc": "Rule MPC",
    }
    rows: list[list[str]] = []
    for baseline in names:
        for metric, label, unit, _, _ in METRICS:
            record = one_row(external, baseline=baseline, metric=metric)
            display = f"{label} ({unit})" if unit else label
            rows.append([
                names[baseline],
                display,
                signed(float(record["learned_mean"]) * METRIC_SPEC[metric][2], METRIC_SPEC[metric][3]),
                signed(float(record["baseline_mean"]) * METRIC_SPEC[metric][2], METRIC_SPEC[metric][3]),
                effect(record, "delta_learned_minus_baseline", metric),
                p_value(record["paired_signflip_p_holm"]),
            ])
    headers = ["Baseline", "Outcome", "FreqDuet mean", "Baseline mean", "Delta [95% CI]", "Holm p"]
    return markdown_table(headers, rows), latex_table(headers, rows, "llrrrr")


def relative_link(source: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(source, output_dir)).as_posix()


def copy_figure_previews(package_dir: Path, out_dir: Path) -> list[str]:
    relative_paths = []
    for index, stem in FIGURE_STEMS.items():
        filename = f"fig{index}_protocol_v6_{stem}.png"
        source = package_dir / "figures" / filename
        require(source.is_file(), f"missing Figure {index} preview")
        relative = Path("figures") / filename
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        relative_paths.append(relative.as_posix())
    return relative_paths


def copy_submission_figures(package_dir: Path, submission_dir: Path) -> list[str]:
    relative_paths = []
    for index, stem in FIGURE_STEMS.items():
        filename = f"fig{index}_protocol_v6_{stem}.pdf"
        source = package_dir / "figures" / filename
        require(source.is_file(), f"missing publication figure: {filename}")
        target = submission_dir / filename
        shutil.copyfile(source, target)
        relative_paths.append((Path("trc_submission") / filename).as_posix())
    return relative_paths


def submission_markdown(parts: Iterable[str]) -> str:
    manuscript = "\n\n".join(part.strip() for part in parts) + "\n"
    for index, stem in FIGURE_STEMS.items():
        manuscript = manuscript.replace(
            f"figures/fig{index}_protocol_v6_{stem}.png",
            f"fig{index}_protocol_v6_{stem}.pdf",
        )
    return manuscript


def metadata_yaml(data: dict[str, Any]) -> str:
    abstract = abstract_body(data).strip()
    abstract_lines = "\n".join(f"  {line}" if line else "" for line in abstract.splitlines())
    keywords = "\n".join(f"  - {json.dumps(keyword)}" for keyword in KEYWORDS)
    return f"title: {json.dumps(TITLE)}\nabstract: |\n{abstract_lines}\nkeywords:\n{keywords}\n"


def highlights_text() -> str:
    highlights = (
        "A causal demand filter assigns slow and fast signals to different bus controls.",
        "The controller passed a preregistered short-training regularity gate.",
        "Long training did not retain the registered regularity gain.",
        "Fixed headway gave shorter journeys and fewer fleet readiness delays.",
    )
    require(all(len(item) <= 85 for item in highlights), "Elsevier highlight exceeds 85 characters")
    return "\n".join(f"- {item}" for item in highlights) + "\n"


def trc_readme_text() -> str:
    return """# Anonymous TRC Working Bundle

This flat directory is generated from the frozen Protocol V6 evidence package.
On a server with the isolated paper toolchain, run:

```bash
./build.sh
```

The command writes `manuscript.tex`, `manuscript.pdf`, `supplementary.tex`, and
`supplementary.pdf`. A successful build means that the prose, citations,
tables, and figure paths compile. It does not override the failed V9
long-training gate. Before submission, replace anonymous metadata and re-check
the current TRC submission portal requirements.
`highlights.txt` is the source text for four Elsevier-length bullets; convert it
to the portal's required upload format at final submission.
"""


def write_submission_bundle(
    package_dir: Path,
    out_dir: Path,
    data: dict[str, Any],
    sources: dict[str, str],
    manuscript_body: str,
    supplementary: str,
) -> list[str]:
    submission_dir = out_dir / "trc_submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "manuscript.tex",
        "manuscript.pdf",
        "manuscript.log",
        "manuscript.aux",
        "manuscript.bbl",
        "manuscript.blg",
        "supplementary.tex",
        "supplementary.pdf",
        "supplementary.log",
        "supplementary.aux",
    ):
        (submission_dir / stale).unlink(missing_ok=True)

    supplementary_prefix = "# Supplementary Material\n\n"
    require(
        supplementary.startswith(supplementary_prefix),
        "supplementary title is missing",
    )
    files = {
        "manuscript_body.md": manuscript_body,
        "supplementary_body.md": supplementary[len(supplementary_prefix):],
        "metadata.yaml": metadata_yaml(data),
        "references.bib": sources["references.bib"],
        "elsarticle-template.tex": sources["elsarticle-template.tex"],
        "supplementary-template.tex": sources["supplementary-template.tex"],
        "build.sh": sources["build_trc_submission.sh"],
        "README.md": trc_readme_text(),
        "highlights.txt": highlights_text(),
    }
    outputs = []
    for name, content in files.items():
        target = submission_dir / name
        target.write_text(content)
        outputs.append((Path("trc_submission") / name).as_posix())
    (submission_dir / "build.sh").chmod(0o755)
    outputs.extend(copy_submission_figures(package_dir, submission_dir))
    return outputs


def source_package_label(package_dir: Path) -> str:
    resolved = package_dir.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def manuscript_methods(method: dict[str, Any], figure_path: str) -> str:
    actions = ", ".join(f"{value:g}" for value in method["action_bins_s"])
    context = ", ".join(f"`{item}`" for item in method["lower_context_features"])
    cost_weights = method["service_cost_weights"]
    return f"""# Methods

## Study question and control setting

FreqDuet studies whether frequency structure in an exogenous demand stream can
be aligned with authority in an asynchronous hierarchical controller. The
upper policy changes an executable target-headway plan at dispatch events; the
lower policy chooses holding at station-arrival events. One upper decision can
therefore span many lower decisions. The current paper controller is
`{PAPER_CONTROLLER}` under `{PROTOCOL}`. It is the exact naming alias of the
compact APC/AVL, weight-two controller that passed the preregistered V8 gate.

![Figure 1. Causal frequency-to-authority architecture.]({figure_path})

## Simulation environment and common random numbers

The simulator advances in 1-s steps on one bidirectional corridor with 22
physical stops (two terminals and 20 intermediate stops) and 42 directed
inter-stop links. Each direction is 10.5 km long, comprising 21 links of 500 m.
The input timetable contains 262 trips, split equally between directions.
Service operates from
{method['service_start_hour']:02d}:00 to {method['service_end_hour']:02d}:00
with a {method['clearance_time_s'] / 3600:g}-hour clearance period, all
scheduled trips, a fixed pool of {method['fleet_size']} physical vehicles, and
a capacity of 50 passengers per vehicle.

Passenger demand comes from a 20-origin by 14-hour by 20-destination historical
OD-intensity table. Every 20 s, the simulator draws an independent Poisson count
for each active OD cell and assigns each generated passenger a uniform arrival
time within that window. A passenger remains latent until its assigned arrival
time has elapsed. Each service-hour intensity is multiplied by a
`Normal(1, 0.15)` draw clipped to `[0.3, 2.0]`; the historical peak profile is
also shifted by -1, 0, or +1 hour with probabilities 0.2, 0.6, and 0.2. Segment
speed limits update every 300 s by adding Gaussian variation with standard
deviation 1.5 to the corresponding hourly route-history value, clipping the
draw to `[2, 15]`, and applying the segment maximum. An inherited corridor
calibration multiplies reverse-direction OD intensities at X13--X15 by 0.4 for
every policy.

All exogenous draws are supplied by a policy-independent scenario tape keyed by
evaluation seed and process identity. Passenger counts and arrival times,
hourly demand multipliers, peak shifts, and fixed-clock route-speed streams
therefore remain aligned across paired policies even when their action and
learning call sequences differ. V8 and V9 use identical tapes within each
policy pair. The public AFC/APC data are used only for the separate realism
audit and do not calibrate the evaluated policy.

## Causal harmonic demand decomposition

Observed APC arrivals are accumulated in {method['bin_sec']:g}-s bins. For bin
`k`, the harmonic basis contains an intercept, a within-day linear trend, and
`K={method['fourier_k']}` sine/cosine pairs over a
{method['harmonic_period_s'] / 3600:g}-h period. Historical OD intensities fit a
ridge-regularized prior for `log(1 + arrival rate)` with ridge
{method['harmonic_ridge']:g} and initial covariance
{method['harmonic_prior_var']:g}. Recursive least squares with forgetting
factor {method['harmonic_forgetting']:g} then updates that prior only after the
current observation bin closes. With
basis `phi_k`, coefficients `theta_k`, and pre-update prediction
`lambda_hat_(k|k-1)`, the high-frequency innovation is

```text
r_k = y_k - lambda_hat_(k|k-1).
```

The low-frequency state is the updated nonnegative harmonic rate, its slope,
and its {method['forecast_horizon_s'] / 60:g}-min forecast. The residual and
its exponentially smoothed energy form the high-frequency state. Decisions at
the start of a bin cannot observe arrivals later in that bin. This ordering,
rather than a full-day transform, is the operational no-leakage guarantee.

## Frequency-to-authority allocation

The upper policy receives global low-frequency level, slope and forecast,
together with a scalar high-frequency energy summary and low-frequency OD
structure summaries. The lower policy receives station-direction residual,
residual change, local and global residual energy, the previous holding action,
and compact same-time APC/AVL context: {context}. The scalar energy summary
alerts the upper layer to volatility without giving it the local residual that
drives holding.

The current controller does not use the historical promotion, leakage-penalty,
or legacy causal holding-guard branches. Those mechanisms were development
variants and are not part of the evaluated V6 policy. Consequently, the paper
claim is the behavior of the complete current controller, not an isolated
effect of promotion, leakage regularization, or guard removal.

## Executable upper timetable

The upper ensemble actor produces a headway adjustment bounded to
[{method['upper_delta_min_s']:g}, {method['upper_delta_max_s']:g}] s. Every
{method['replan_interval_s'] / 60:g} min, the timetable planner maps that action
to an exact terminal headway curve over a
{method['planning_horizon_s'] / 60:g}-min horizon. The
`{method['headway_budget_mode']}` projection conserves the cumulative headway
budget over each closed replanning window, preventing hidden phase drift.
Planned launch times are executable: an actual departure occurs no earlier
than both vehicle readiness and the scheduled launch time. Planned and actual
terminal times are logged separately, and terminal shifts are bounded to
[{method['terminal_shift_min_s']:g}, {method['terminal_shift_max_s']:g}] s.

## Discrete lower holding and regularity reward

At each eligible station arrival, the lower categorical ensemble policy
chooses holding from `{{{actions}}}` s. Training samples from the policy;
frozen evaluation uses its deterministic output. The chosen action is sent to
the environment without a post-policy projection. The lower state includes the
analytic balancing target derived from a matched predecessor departure and a
same-time AVL estimate of the following vehicle.

Let `g_f` be the pre-action forward departure gap, `g_b` the same-time
follower gap, `h` the executable target headway, and `a` the sampled hold. A
hold predicts gaps `g_f + a` and `max(g_b - a, 0)`. For tolerance
`tau={method['regularity_tolerance_fraction']:.2f}`, define

```text
q(g,h) = max(|g-h|/h - tau, 0)^2
L(g_f,g_b,h) = 0.5 * [q(g_f,h) + q(g_b,h)].
```

The lower reward receives
`{method['regularity_reward_weight']:g} * clip(L_before - L_after, -{method['regularity_cost_cap']:g}, {method['regularity_cost_cap']:g})`.
All gaps are frozen before the action. Missing predecessor or follower evidence
adds zero regularity reward and is logged rather than imputed from future
vehicle states.

## Learning architecture

Both levels use off-policy soft actor-critic variants
[@haarnoja2018soft]. The upper
`{method['upper_algorithm']}` uses a pessimistic
{method['upper_ensemble_size']}-critic ensemble, discount
{method['upper_discount']:g}, a {method['upper_hidden_dim']}-unit hidden
representation, batch size {method['upper_batch_size']}, and
{method['upper_updates_per_episode']} updates per episode. The lower
`{method['lower_algorithm']}` uses a pessimistic
{method['lower_ensemble_size']}-critic ensemble, discount
{method['lower_discount']:g}, a {method['lower_hidden_dim']}-unit hidden
representation, batch size {method['lower_batch_size']}, and
{method['lower_updates_per_episode']} updates per episode. Both learning rates
are `{method['upper_learning_rate']:g}`; the lower dual learning rate is
`{method['lower_dual_learning_rate']:g}`. The lower policy's learned constraint
cost is optimized through a Lagrange multiplier, following the constrained-RL
formulation [@miryoosefi2019constraints]. The upper policy begins after a
{method['upper_warmup_episodes']}-episode lower
warm-up. V8 trains for 40 episodes and evaluates checkpoint 39; V9 trains
without policy or critic freezing for 200 episodes and evaluates checkpoint
199. Both upper and lower actors are deterministic during frozen evaluation,
and the evaluator rejects any change in deployment state across evaluation
seeds.

## External comparators

Three non-learned comparators use the same V6 environment, fixed 12-vehicle
pool, timetable, evaluation seeds, scenario tapes, and exact terminal-release
semantics as the learned controller. `fixed_headway` installs a 360-s terminal
headway in both directions and commands zero intermediate holding.
`rule_holding` uses the same 360-s terminal schedule and applies

```text
a = clip(360 - g_f, 0, 60),
```

where `g_f` is the observed forward headway in seconds. `rule_mpc` uses that
same lower holding law and re-evaluates a 60-candidate time-of-day headway grid
at dispatch events. Peak candidates are `{{240, 300, 360, 420, 480}}` s,
off-peak candidates are `{{360, 480, 600, 720}}` s, and transition candidates
are `{{300, 360, 420}}` s. Given a seeded episode-level demand proxy `d`, with
`d ~ clip(Normal(1, 0.15), 0.3, 2.0)`, its slot-specific surrogate is

```text
H_ideal = clip(360 / d, 240, 600)
J(H) = H / 2 + 0.001 max(0, H - H_ideal)^2
       + 5 max(0, 6000 / H - 12)^2.
```

The minimizing candidate is converted to an exact launch sequence before
simulation. This rule-MPC is a transparent low-fidelity comparator, not a claim
to represent an optimally tuned or simulator-aware MPC. The fixed-headway
policy is the strong external comparator in this study.

## Outcomes

The primary passenger endpoint is restricted total journey time per generated
passenger. Waiting is censored at the evaluation horizon for passengers not
yet boarded; in-vehicle and total journey time are censored at that horizon for
passengers not yet arrived. Secondary outcomes are restricted waiting time,
restricted in-vehicle time, unserved-passenger rate, headway coefficient of
variation, realized vehicle and passenger holding, fleet-denial measures,
terminal execution error, trip completion, and restricted service cost.
Headway CV is the standard deviation divided by the mean over valid recorded
headway events. With restricted waiting `W_R` in minutes, peak fleet `F`, fixed
fleet budget `N`, headway CV `H`, unserved fraction `U`, and trip-completion
fraction `Q`, the secondary scalar is

```text
C_R = W_R / 10
    + max(F - N, 0)^2 / N
    + H
    + {cost_weights['unserved']:g} U
    + {cost_weights['incomplete_service']:g} (1 - Q).
```

The evaluated fixed-pool environment makes the overshoot term zero; this
scalar does not directly charge holding or a delayed-readiness denial that is
later retried. It is therefore a secondary summary and does not replace the
passenger and physical outcomes.

## Evaluation and inference

V8 is a preregistered independent 40-episode confirmation with six training
seeds crossed with four untouched evaluation seeds (24 paired rollouts per
policy). V9 is a separately preregistered 200-episode robustness test with
eight new training seeds crossed with eight new evaluation seeds (64 paired
rollouts per policy). The Protocol V6 reference is `{REFERENCE}`. Both policies
disable the legacy holding guard; the current policy additionally has compact
APC/AVL context and the two-sided regularity objective, so their difference is
a combined-policy contrast.

Uncertainty uses a crossed bootstrap over training and evaluation seeds while
sharing each evaluation-seed resample across paired policies. Two-sided
sign-flip tests operate on training-seed mean differences, with Holm correction
within each metric family. Lower values favor FreqDuet for every reported
outcome. For each external comparator, its eight evaluation-seed realizations
are crossed with the eight V9 learned training seeds, yielding 64 paired rows;
the crossed analysis retains training seed as the independent policy-training
unit. The V8 and V9 estimates are kept separate and are never pooled.
"""


def manuscript_results(
    data: dict[str, Any],
    table1: str,
    table2: str,
    out_dir: Path,
) -> str:
    v8 = data["v8"]
    v9 = data["v9"]
    external = data["external"]

    def pair(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
        return one_row(rows, metric=metric)

    v8_journey = pair(v8, "passenger_journey_min")
    v8_cv = pair(v8, "headway_cv")
    v9_journey = pair(v9, "passenger_journey_min")
    v9_cv = pair(v9, "headway_cv")
    v9_hold = pair(v9, "holding_s_per_trip")
    v9_denied = pair(v9, "denied_trip_rate")
    fixed_journey = one_row(external, baseline="fixed_headway", metric="passenger_journey_min")
    fixed_cv = one_row(external, baseline="fixed_headway", metric="headway_cv")
    fixed_cost = one_row(external, baseline="fixed_headway", metric="restricted_service_cost")
    fixed_hold = one_row(external, baseline="fixed_headway", metric="holding_s_per_trip")
    fixed_denied = one_row(external, baseline="fixed_headway", metric="denied_trip_rate")
    rule_journey = one_row(external, baseline="rule_holding", metric="passenger_journey_min")
    mpc_journey = one_row(external, baseline="rule_mpc", metric="passenger_journey_min")

    figures = {
        index: relative_link(
            out_dir / "figures" / f"fig{index}_protocol_v6_{stem}.png", out_dir
        )
        for index, stem in FIGURE_STEMS.items()
    }

    return f"""# Results

## Independent confirmation at 40 episodes

V8 passed the registered effect/no-harm gate for the complete current policy
(Fig. 2; Table 1). Relative to the Protocol V6 reference, headway CV changed by
{effect(v8_cv, 'delta_candidate_minus_reference', 'headway_cv')}. Its
crossed-bootstrap interval excluded zero, whereas the Holm-adjusted
training-seed sign-flip result was
$p={p_value(v8_cv['paired_signflip_p_holm'])}$. Restricted
passenger journey changed by
{effect(v8_journey, 'delta_candidate_minus_reference', 'passenger_journey_min')}
min. The latter interval crossed zero but satisfied the preregistered journey
no-harm margin. Thus V8 is gate-positive under its preregistered criteria; it
is not a familywise-significant effect at 0.05 and does not establish a
passenger-journey benefit.

![Figure 2. Independent confirmation and long-training robustness.]({figures[2]})

**Table 1. Current policy minus the Protocol V6 reference.** Values are paired
mean differences with crossed-bootstrap 95% confidence intervals. Lower is
better. Holm-adjusted sign-flip p-values use training-seed mean differences.

{table1}

## Long-training robustness was not confirmed

At 200 episodes, the same policy improved restricted journey by
{effect(v9_journey, 'delta_candidate_minus_reference', 'passenger_journey_min')}
min relative to the reference. The headway-CV difference was
{effect(v9_cv, 'delta_candidate_minus_reference', 'headway_cv')}. Seven of
eight training-seed CV differences were negative, but the interval included
zero and the mean improvement did not reach the registered 0.02 threshold.
V9 therefore returned `longtrain_not_confirmed`; the favorable journey result
cannot be relabelled as confirmation of the registered regularity effect.

## External baselines reveal a passenger-regularity trade-off

The source-identical V9 comparison (Fig. 3; Table 2) shows that FreqDuet had
lower headway CV than fixed headway,
{effect(fixed_cv, 'delta_learned_minus_baseline', 'headway_cv')}, and lower
restricted service cost,
{effect(fixed_cost, 'delta_learned_minus_baseline', 'restricted_service_cost')}.
However, restricted journey was higher by
{effect(fixed_journey, 'delta_learned_minus_baseline', 'passenger_journey_min')}
min. FreqDuet also used
{effect(fixed_hold, 'delta_learned_minus_baseline', 'holding_s_per_trip')} more
holding seconds per launched trip and had a
{effect(fixed_denied, 'delta_learned_minus_baseline', 'denied_trip_rate')}
percentage-point higher denied-trip rate. All trips were eventually launched
and completed in the aggregated learned-policy results, so this denial measure
captures delayed fleet readiness rather than permanent trip cancellation.
Because the restricted service-cost scalar does not directly charge holding or
retried readiness denials, its favorable difference cannot be interpreted as
passenger-time or fleet-readiness superiority.

FreqDuet reduced restricted journey relative to rule holding by
{effect(rule_journey, 'delta_learned_minus_baseline', 'passenger_journey_min')}
min and relative to rule MPC by
{effect(mpc_journey, 'delta_learned_minus_baseline', 'passenger_journey_min')}
min. The supported external conclusion is therefore narrower than universal
superiority: FreqDuet reduced restricted journey relative to the two rules and
produced more regular service than fixed headway, while fixed headway remained
better for passenger journey and fleet-readiness burden.

![Figure 3. V9 external-baseline trade-off.]({figures[3]})

**Table 2. FreqDuet minus external baseline under V9.** Values are paired mean
differences with crossed-bootstrap 95% confidence intervals. Lower is better.
The complete outcome and adjusted-test table is in the Supplementary Material.

{table2}

## Physical execution audit

Relative to the Protocol V6 reference, V9 reduced realized holding by
{effect(v9_hold, 'delta_candidate_minus_reference', 'holding_s_per_trip')} s
per launched trip and the denied-trip rate by
{effect(v9_denied, 'delta_candidate_minus_reference', 'denied_trip_rate')}
percentage points (Fig. 4). These are full-policy differences, not isolated
effects of the regularity reward. Against fixed headway, however, the current
policy used
{effect(fixed_hold, 'delta_learned_minus_baseline', 'holding_s_per_trip')} more
holding seconds per launched trip and increased the denied-trip rate by
{effect(fixed_denied, 'delta_learned_minus_baseline', 'denied_trip_rate')}
percentage points. All trips were eventually launched and completed in the
aggregated learned-policy results, so denial records delayed fleet readiness
rather than permanent trip cancellation.

![Figure 4. Paired physical outcomes.]({figures[4]})

## External data support demand-shape realism only

The FreqDuet OD input has a morning peak similar in timing to the bounded MTA
AFC subset, while its normalized hourly profile differs from the Halifax APC
subset (Fig. 5). The balanced audit contains 39 complete MTA station-complex
days (936 rows) and seven complete Halifax routes across 37 route-days (979
rows). Because systems, dates, sampling units, and measurement processes are
unmatched, these comparisons are descriptive checks of demand-shape
plausibility. They are not same-day calibration, route-family policy tests, or
field-effect estimates.

![Figure 5. External passenger-count demand-shape audit.]({figures[5]})
"""


def abstract_body(data: dict[str, Any]) -> str:
    v8_cv = one_row(data["v8"], metric="headway_cv")
    v8_journey = one_row(data["v8"], metric="passenger_journey_min")
    v9_cv = one_row(data["v9"], metric="headway_cv")
    v9_journey = one_row(data["v9"], metric="passenger_journey_min")
    fixed_journey = one_row(
        data["external"], baseline="fixed_headway", metric="passenger_journey_min"
    )
    return f"""Bus timetable planning and station-level holding operate at
different temporal and physical scales, creating an information-allocation
problem for a learned hierarchy.
FreqDuet estimates demand causally from completed APC bins, sends a
harmonic low-frequency state to an executable upper headway planner, and sends
station-local innovations plus compact APC/AVL context to a discrete lower
holding policy. In an independent 40-episode confirmation with 24 paired
rollouts, the complete current controller changed headway coefficient of
variation by {effect(v8_cv, 'delta_candidate_minus_reference', 'headway_cv')}
relative to a same-protocol reference. The bootstrap interval excluded zero,
whereas the Holm-adjusted training-seed sign-flip result was
$p={p_value(v8_cv['paired_signflip_p_holm'])}$. Restricted passenger journey changed by
{effect(v8_journey, 'delta_candidate_minus_reference', 'passenger_journey_min')}
min and satisfied the registered no-harm condition. In a separate 200-episode,
64-pair robustness test, journey improved by
{effect(v9_journey, 'delta_candidate_minus_reference', 'passenger_journey_min')}
min (Holm-adjusted $p={p_value(v9_journey['paired_signflip_p_holm'])}$), but the headway effect weakened to
{effect(v9_cv, 'delta_candidate_minus_reference', 'headway_cv')} and failed the
registered long-training gate. Against fixed headway, FreqDuet was more regular
but increased passenger journey by
{effect(fixed_journey, 'delta_learned_minus_baseline', 'passenger_journey_min')}
min. The results support a registered gate-positive short-training regularity
signal and expose its training-horizon and passenger-service limits; they do not establish
long-run regularity confirmation, passenger-service superiority over fixed
headway, or an isolated causal effect of frequency separation.
"""


def abstract_text(data: dict[str, Any]) -> str:
    keywords = "; ".join(KEYWORDS)
    return f"# {TITLE}\n\n## Abstract\n\n{abstract_body(data)}\n\n**Keywords:** {keywords}\n"


def supplementary_text(
    data: dict[str, Any],
    full_external: str,
    main_table: str,
) -> str:
    method = data["method"]
    decisions = data["decisions"]
    decision_rows = [
        [
            row["phase"],
            row["controller"],
            row["train_seeds"],
            row["evaluation_seeds"],
            row["paired_rollouts"],
            row["decision"],
            row["claim_eligible"],
        ]
        for row in decisions
    ]
    decision_table = markdown_table(
        ["Phase", "Controller", "Train seeds", "Eval seeds", "Pairs", "Decision", "Eligible"],
        decision_rows,
    )
    lineage = "\n".join(f"{index}. `{item}`" for index, item in enumerate(method["config_lineage"], 1))

    return f"""# Supplementary Material

## S1. Frozen evidence and decision ledger

The current paper package binds the gate-positive V8 decision and failed V9
long-training gate. They use disjoint training and evaluation seeds and are not
pooled. The V8 decision label records its preregistered effect/no-harm gate; it
does not imply a familywise-significant result at 0.05.

{decision_table}

The term `confirmed_main` identifies the V8-selected configuration; it does not
mean that V9 passed. The machine-readable package remains
`submission_ready: false` with blocker `v9_longtrain_not_confirmed`.

## S2. Exact current-controller configuration

The current controller resolves through the following inheritance chain:

{lineage}

Key resolved settings are: harmonic historical prior; 60-s causal bins;
four Fourier harmonics over 14 h; 30-min low-frequency forecast; 15-min upper
replanning over 45 min; rolling zero-sum V6 headway budget; executable terminal
dispatch; low-frequency upper and high-frequency lower authority; compact
same-time APC/AVL lower context; previous-action state; holding actions
`{{0, 5, 10, 15, 20, 30, 45}}` s; and a weight-two pre-action two-sided
regularity reward. The legacy holding guard, promotion, and leakage penalty are
disabled.

## S3. Causal and physical contract

Passenger arrivals become visible only after their within-bin arrival times.
APC counts enter the frequency tracker only when a complete 60-s bin closes.
The lower regularity tuple freezes the matched predecessor departure, same-time
follower AVL estimate, target headway, and action before transition settlement.
The categorical action is executed without post-policy clipping. Commanded and
realized holding are recorded separately. The upper timetable materializes
future launch times once; cached reuse is read-only; each closed rolling budget
block conserves its headway adjustment; and actual launch cannot precede either
vehicle readiness or the executable scheduled time.

## S4. Complete V8 and V9 policy-reference outcomes

{main_table}

All entries are current policy minus `{REFERENCE}`. The V8 row corresponds to
the source config `{SOURCE_CANDIDATE}` and the V9 row to its exact paper alias
`{PAPER_CONTROLLER}`. Both arms disable the legacy holding guard. The contrast
combines compact APC/AVL context with the incremental regularity objective.

## S5. Complete V9 external-baseline outcomes

{full_external}

Means and differences use the same outcome units shown in each row. The
denied-trip rate is the fraction of scheduled trips denied at least once by
the fixed-pool readiness check; retry duration is reported separately in the
source artifacts. A launch rate of one does not imply a zero denial rate.

## S6. Statistical procedure

Each train-seed policy is evaluated on every registered evaluation seed under
common random numbers. Crossed-bootstrap intervals resample training seeds and
a shared set of evaluation seeds, preserving policy pairing. Sign-flip tests
operate on train-seed mean deltas and therefore target conditional
training-seed inference rather than the crossed population represented by the
bootstrap. Holm adjustment is performed across compared methods separately for
each outcome. No V8 and V9 estimate is pooled, and no failed gate is rescued by
a favorable secondary endpoint.

The V8 gate required a headway-CV improvement of at least 0.02 against the
Protocol V6 reference, at least 0.01 against compact context alone, passenger
journey within +0.15 min of both references, complete paired rollouts, at least
50% same-time follower coverage, zero execution adjustment, and preservation
of holding and denied-dispatch gains. V9 reused those gates and additionally
required a headway-CV interval below zero, a journey CI upper bound no larger
than +0.15 min, and negative train-seed CV differences for at least 75% of
training seeds. V9 met the latter two added directional/no-harm conditions but
failed the inherited CV magnitude gates and CI-exclusion requirement.

## S7. Frequency-claim boundary and negative development evidence

The current confirmatory package does not contain a same-stage NoFreq,
RawHistory, AllFreq, swapped-layer, promotion, or leakage ablation. Earlier V6
engineering screens used some of these controls, but they were exploratory,
incomplete, or tied to superseded controller semantics. They cannot be used as
confirmatory evidence that frequency separation itself caused the V8 effect.
Likewise, V28-V32 counterfactual value/rank/margin planners all failed their
registered development gates and were not promoted. These negative results
define the stopping decision: the manuscript uses the frozen V6 controller and
does not tune another gate on the same development contexts.

## S8. External-data provenance

Figure 5 uses separately normalized demand shapes from the local FreqDuet OD
input, 39 complete station-complex days selected from a bounded public MTA AFC
cache, and seven complete routes spanning 37 route-days selected from a bounded
Halifax APC cache. Incomplete pagination fragments were excluded before
aggregation. MTA Bus Time is used only as route/stop and AVL audit data, not APC
or onboard load. MBTA boarding, alighting, and load data provide separate
calibration targets but are not a same-day matched calibration of the simulated
network. API credentials are not stored in the repository or paper package.

## S9. Reproducibility boundary

The paper evidence directory tracks the small V8/V9 CSV and JSON artifacts,
exact seed contracts, source commits, and resolved config lineage. Checkpoints
and full training logs remain on the HPC filesystem and are intentionally not
part of the manuscript package. A build validates the frozen package before
rendering any text or table. A successful build verifies consistency; it does
not change the scientific decision or create a field-deployment claim.
"""


def readme_text() -> str:
    return f"""# Protocol V6 Manuscript Assembly

This directory is generated from the frozen Protocol V6 evidence package by
`freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py`.

From the repository root, rebuild it with:

```bash
python FreqDuet/freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py
```

## Contents

- `introduction.md` and `related_work.md`: source-grounded positioning.
- `methods.md`: current method and evaluation protocol.
- `results.md`: evidence-bound main results and two main tables.
- `discussion.md` and `conclusion.md`: interpretation and claim boundaries.
- `availability.md`: evidence, public-data, and archive-status statement.
- `manuscript.md`: complete assembled article draft.
- `supplementary.md`: full outcomes, configuration lineage, statistical
  procedure, negative-result boundary, and external-data provenance.
- `references.bib`: verified working bibliography.
- `terminology.md`: canonical paper terms and prohibited conflations.
- `literature_verification.md`: primary-record bibliography audit.
- `journal_target.md`: target-journal fit and formatting decision.
- `trc_submission/`: flat anonymous `elsarticle` working bundle with separate
  manuscript and Supplementary Material builds.
- `tables/`: standalone Markdown and LaTeX table fragments.
- `figures/`: portable review PNGs for Figures 1-5; publication-format exports
  remain in the frozen evidence package.
- `figure_captions.md`: assembled captions for Figures 1-5.
- `assembly_manifest.json`: source and output inventory.

## Scientific status

The manuscript uses `{PAPER_CONTROLLER}` as the current best controller. V8
passed its registered 40-episode effect/no-harm gate, although its
Holm-adjusted training-seed sign-flip result is $p=0.125$. V9 did not confirm
the registered 200-episode regularity gate. The V9
fixed-headway comparison is a trade-off, not a passenger-journey superiority
result. The evidence status therefore remains `submission_ready: false`; adding
editorial sections and a journal template does not override that scientific
decision. Author metadata and the current portal-specific submission fields
remain pre-submission tasks.
"""


def figure_caption_sections(package_dir: Path) -> dict[int, str]:
    sections: dict[int, str] = {}
    for name in ("supporting_captions.md", "captions.md"):
        source = (package_dir / "figures" / name).read_text()
        for match in re.finditer(
            r"^## Figure (\d+)\b.*?(?=^## Figure \d+\b|\Z)",
            source,
            flags=re.MULTILINE | re.DOTALL,
        ):
            index = int(match.group(1))
            require(index not in sections, f"duplicate Figure {index} caption")
            sections[index] = match.group(0).strip()
    require(set(sections) == set(range(1, 6)), "expected captions for Figures 1-5")
    return sections


def figure_caption_text(section: str) -> str:
    heading, body = section.split("\n", maxsplit=1)
    require(" | " in heading, f"unexpected figure caption heading: {heading}")
    title = heading.split(" | ", maxsplit=1)[1].strip()
    description = " ".join(body.split())
    return f"{title}. {description}"


def embed_figure_captions(
    markdown: str,
    sections: dict[int, str],
    indices: Iterable[int],
) -> str:
    rendered = markdown
    for index in indices:
        caption = figure_caption_text(sections[index]).replace("]", r"\]")
        pattern = re.compile(
            rf"!\[Figure {index}\.[^\]]*\]\(([^)]+)\)"
        )
        rendered, replacements = pattern.subn(
            lambda match: (
                f"![{caption}]({match.group(1)})"
                f"{{#fig:protocol-v6-{index}}}"
            ),
            rendered,
        )
        require(replacements == 1, f"expected one Figure {index} placeholder")
    return rendered


def ordered_figure_captions(sections: dict[int, str]) -> str:
    return "# Protocol V6 Figure Captions\n\n" + "\n\n".join(
        sections[index] for index in range(1, 6)
    ) + "\n"


def build_manuscript(package_dir: Path, out_dir: Path) -> dict[str, Any]:
    data = validate_package(package_dir)
    sources = read_editorial_sources()
    caption_sections = figure_caption_sections(package_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    main_md, main_tex = main_confirmation_table(data["v8"], data["v9"])
    external_md, external_tex = external_main_table(data["external"])
    external_full_md, external_full_tex = external_full_table(data["external"])
    figure_outputs = copy_figure_previews(package_dir, out_dir)

    method_figure = relative_link(
        out_dir / "figures" / "fig1_protocol_v6_method.png", out_dir
    )
    methods = embed_figure_captions(
        manuscript_methods(data["method"], method_figure),
        caption_sections,
        (1,),
    )
    results = embed_figure_captions(
        manuscript_results(data, main_md, external_md, out_dir),
        caption_sections,
        (2, 3, 4, 5),
    )
    abstract = abstract_text(data)
    supplementary = supplementary_text(data, external_full_md, main_md)
    article_sections = (
        sources["introduction.md"],
        sources["related_work.md"],
        methods,
        results,
        sources["discussion.md"],
        sources["conclusion.md"],
        sources["availability.md"],
    )
    article_body = "\n\n".join(section.strip() for section in article_sections) + "\n"
    require_resolved_citations(article_body, sources["references.bib"])
    trc_body = submission_markdown(article_sections)

    outputs = {
        "README.md": readme_text(),
        "introduction.md": sources["introduction.md"],
        "related_work.md": sources["related_work.md"],
        "methods.md": methods,
        "results.md": results,
        "discussion.md": sources["discussion.md"],
        "conclusion.md": sources["conclusion.md"],
        "availability.md": sources["availability.md"],
        "manuscript.md": abstract.rstrip() + "\n\n" + article_body,
        "supplementary.md": supplementary,
        "references.bib": sources["references.bib"],
        "terminology.md": sources["terminology.md"],
        "literature_verification.md": sources["literature_verification.md"],
        "journal_target.md": sources["journal_target.md"],
        "tables/table1_confirmation_and_robustness.md": main_md + "\n",
        "tables/table1_confirmation_and_robustness.tex": main_tex,
        "tables/table2_external_tradeoff.md": external_md + "\n",
        "tables/table2_external_tradeoff.tex": external_tex,
        "tables/table_s1_external_full.md": external_full_md + "\n",
        "tables/table_s1_external_full.tex": external_full_tex,
        "figure_captions.md": ordered_figure_captions(caption_sections),
    }
    for relative, content in outputs.items():
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    submission_outputs = write_submission_bundle(
        package_dir,
        out_dir,
        data,
        sources,
        trc_body,
        supplementary,
    )

    manifest = {
        "assembly_version": "freqduet-protocol-v6-manuscript-v4",
        "protocol": PROTOCOL,
        "target_journal": "Transportation Research Part C: Emerging Technologies",
        "working_template": "elsarticle-preprint-authoryear",
        "paper_controller": PAPER_CONTROLLER,
        "source_candidate": SOURCE_CANDIDATE,
        "reference": REFERENCE,
        "v8_status": "primary_confirmed",
        "v9_status": "longtrain_not_confirmed",
        "submission_ready": False,
        "submission_blocker": "v9_longtrain_not_confirmed",
        "source_package": source_package_label(package_dir),
        "outputs": sorted([*outputs, *figure_outputs, *submission_outputs]),
    }
    (out_dir / "assembly_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_manuscript(args.package_dir.resolve(), args.out_dir.resolve())
    print(json.dumps({
        "status": "manuscript_assembly_complete",
        "out_dir": str(args.out_dir.resolve()),
        "output_count": len(manifest["outputs"]) + 1,
        "submission_ready": manifest["submission_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
