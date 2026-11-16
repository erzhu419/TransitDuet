"""Typeset frozen paired-day results without rerunning or resampling them."""

import argparse
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    400: "v400_independent_confirmation/review01",
    409: "v409_holding_dose/review",
    410: "v410_within_day_control/review",
    411: "v411_demand_transfer/review",
    412: "v412_running_speed_transfer/review",
    418: "v418_spacing_confirmation/review",
}
TRANSITION = "v406_cap_10_minus_transition"
FIXED = "v406_cap_10_minus_fixed_cap_10"
HO, JO, CV, RE, PU = (
    "holding_s_per_trip", "journey_s_per_passenger", "cv_squared",
    "readiness_s_per_trip", "public_delay_s_per_day",
)
BINS = [f"bin_{n}_share" for n in (0, 5, 10)]


def load(snapshot):
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    data = {}
    for version, directory in SOURCES.items():
        archived = evidence / f"v{version}.json"
        if snapshot:
            source = ROOT / "experiments" / directory / "aggregate/results.json"
            shutil.copyfile(source, archived)
        data[version] = json.loads(archived.read_text())
    return data


def value(metric, decimals, scale=1):
    mean = f"{metric['mean'] * scale:+.{decimals}f}"
    low, high = (f"{end * scale:+.{decimals}f}" for end in metric["ci95"])
    return rf"\shortstack[r]{{$ {mean} $\\$[{low},{high}]$}}"


def write_table(name, rows):
    lines = [" & ".join(map(str, row)) + r" \\" for row in rows]
    (HERE / "tables" / name).write_text(
        "\n\\addlinespace[3pt]\n".join(lines) + "\n\\bottomrule\n"
    )


def interval_within(metric, margin):
    lo, hi = metric["ci95"]
    return lo >= -margin and hi <= margin


def make_tables(data):
    (HERE / "tables").mkdir(exist_ok=True)
    heavy = data[410]
    regimes = [
        ("High demand ($1.20\\times$)", heavy),
        ("Nominal demand", data[411]["regimes"]["nominal"]),
        ("Higher demand ($1.35\\times$)", data[411]["regimes"]["stress"]),
        ("Normal speed, new days", data[412]["regimes"]["normal"]),
        ("Slower speed, same days", data[412]["regimes"]["slow"]),
    ]
    mixture = data[409]
    placebo = mixture["effects"]["v406_cap_10_matched"]
    capped_gate = mixture["gates"]["v406_cap_10"]
    assert capped_gate["passed"] and capped_gate["dose_matched"]
    assert not mixture["gates"]["v406_single"]["dose_matched"]
    matched, bins, fixed = [], [], []
    for label, block in regimes:
        e = block["effects"][TRANSITION]
        count = len(block["by_day"])
        match = {k: interval_within(e[k], 3 if k == HO else .02) for k in [HO, *BINS]}
        outcomes = {k: e[k]["ci95"][1] <= 0 for k in [CV, RE, PU]}
        outcomes[JO] = e[JO]["ci95"][1] < 0
        assert match == block["matching"] and outcomes == block["outcomes"], label
        assert block["passed"] == (all(match.values()) and all(outcomes.values())), label
        reason = "Pass" if block["passed"] else ("Matching" if not all(match.values())
                                                 else "Circulation")
        metrics = [value(e[k], n) for k, n in ((HO, 3), (JO, 3), (CV, 6), (RE, 3))]
        matched.append([label, count, *metrics, reason])
        bins.append([label, *(value(e[k], 3, 100) for k in BINS), value(e[PU], 1)])
        f = block["effects"][FIXED]
        fixed.append([label, *(value(f[k], n) for k, n in
                               ((HO, 3), (JO, 3), (CV, 6), (RE, 3)))])
    write_table("matched_rows.tex", matched)
    write_table("bin_rows.tex", bins)
    write_table("fixed_rows.tex", fixed)
    effects = heavy["effects"][TRANSITION]
    days = sorted(effects[JO]["by_day"])
    assert len(days) == 16
    per_day = [
        [i, *(effects[k]["by_day"][day] for k in (JO, CV, RE))]
        for i, day in enumerate(days, 1)
    ]
    assert all(all(x < 0 for x in record[1:]) for record in per_day)
    (HERE / "tables" / "high_demand_days.dat").write_text(
        "day journey cv readiness\n"
        + "".join(" ".join(map(str, row)) + "\n" for row in per_day)
    )
    write_table("mixture_rows.tex", [[
        label, *(value(e[k], n) for k, n in ((HO, 3), (JO, 3), (CV, 6), (RE, 3))),
    ] for label, e in [("Capped", placebo),
                       ("Unrestricted", mixture["effects"]["v406_single_matched"])]])

    forecast = data[418]
    assert not forecast["passed"] and len(forecast["independent_days"]) == 16
    write_table("forecast_rows.tex", [
        [label, *(value(forecast["effects"][f"v406_cap_10_hourly_spacing_minus_{arm}"][k], n)
                  for k, n in ((JO, 3), (CV, 6), (RE, 3), (PU, 1)))]
        for label, arm in [
            ("Unguarded pooled", "v406_cap_10"),
            ("Guarded pooled", "v406_cap_10_pooled_spacing"),
        ]
    ])
    budget = data[400]["effects"]
    write_table("budget_rows.tex", [
        [label, *(value(budget[contrast][k], n) for k, n in ((JO, 3), (CV, 6), (RE, 3)))]
        for label, contrast in [
            ("Tighter budget, hand-coded", "manual_tight_minus_default"),
            ("Tighter budget, learned", "K4_tight_minus_default"),
            ("Learned minus hand-coded, tight", "K4_minus_manual_r025"),
            ("Harmonic inputs minus none, default", "K4_minus_K0_r050"),
        ]
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="copy archived result JSON from the checkout")
    make_tables(load(parser.parse_args().snapshot))
