"""Build the manuscript tables from the frozen V500--V502 summaries."""

import json
from pathlib import Path

import numpy as np


PAPER = Path(__file__).resolve().parent
EXPERIMENTS = PAPER.parents[1] / "experiments"
TABLES = PAPER / "tables"
METHODS = {
    "B0_Fixed_NoHold": "Fixed-NoHold",
    "B1_Daganzo2009": "Forward-Hold",
    "B2_Berrebi2018": "Bilateral-Hold",
    "B3_EconMPC": "Rolling-Heuristic",
    "B4_FlatSAC": "Flat-SAC",
    "B5_TransitDuet_NoFreq": "NoFreq-HRL",
    "Abl_RawHistory": "RawHistory",
    "Abl_AllBand": "AllBand",
    "Abl_Swapped": "Swapped",
    "Abl_NoBudget": "NoBudget",
    "Abl_NoPromotion": "NoPromotion",
    "FreqDuet_Aligned": "FreqDuet",
}
CORRIDORS = {
    "mbta_route111_wkdy": "MBTA-111 (data-informed simulation)",
    "tidal_bimodal": "Tidal-Bimodal (synthetic simulation)",
}
REF = "FreqDuet_Aligned"
J = "restricted_journey_s_per_pax"
WAIT = "restricted_wait_s_per_pax"
ONBOARD = "restricted_in_vehicle_s_per_pax"
CV = "raw_cv2"
READY = "readiness_delay_s_per_trip"
EWT = "ewt_s"


def read(path):
    return json.loads(path.read_text())


def rows(name, lines):
    command = "\\Rows" + name.title()
    contents = "\\newcommand{" + command + "}{%\n" + "\n".join(lines) + "\n}\n"
    (TABLES / f"{name}.tex").write_text(contents)


def row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def panel(label, width):
    return rf"\midrule\multicolumn{{{width}}}{{@{{}}l}}{{\textbf{{{label}}}}} \\" + "\n\\midrule"


def interval(values):
    mean, lo, hi = values
    return "$" + rf"{mean:+.2f}\ [{lo:+.2f},\,{hi:+.2f}]$"


def error_bar(mean, sd):
    return "$" + rf"{mean:.2f} \pm {sd:.2f}$"


def pack_contrast(delta, seed_indices, day_indices):
    boot = delta[seed_indices[:, :, None], day_indices[:, None, :]].mean(axis=(1, 2))
    return (float(delta.mean()), *np.quantile(boot, [0.025, 0.975]).tolist())


def v502_tables():
    root = EXPERIMENTS / "v502_pathb_trc_matrix" / "review"
    archive = read(root / "aggregate/results.json")
    rng = np.random.default_rng(20261119)
    nseed = len(archive["seeds"])
    draws = 30_000
    seed_indices = rng.integers(nseed, size=(draws, nseed))
    day_samples = {n: rng.integers(n, size=(draws, n)) for n in (16, 8)}
    nominal, ood, transfer = [], [], []

    for corridor, label in CORRIDORS.items():
        suites = archive["by_corridor"][corridor]
        method_files = {
            key: [read(root / f"classical_{corridor}_{key}/results.json")]
            if key.startswith(("B0_", "B1_", "B2_", "B3_")) else
            [read(root / f"learned_{corridor}_{key}_s{s}/results.json")
             for s in archive["seeds"]]
            for key in METHODS
        }

        def metric(key, suite):
            data = [f["suites"][suite]["per_day"] for f in method_files[key]]
            paired = method_files[REF][0]["suites"][suite]["per_day"]

            def day_identity(days):
                return [(p["day"], p["scenario_tape_id"], p["metrics"]["passengers_generated"],
                         p["metrics"]["physical_buses"]) for p in days]

            for days in data:
                assert day_identity(days) == day_identity(paired), (key, suite)
                assert all(p["metrics"]["trips_launched"] == p["metrics"]["trips_completed"] == 262
                           for p in days), (key, suite)
            out = np.array([[p["metrics"][J] for p in days] for days in data])
            baseline = suites[suite]["methods"][key]["mean"][J]
            assert abs(float(out.mean()) - baseline) < 1e-6, (key, suite)
            return out

        def contrast(key, suite):
            target = metric(REF, suite)
            control = metric(key, suite)
            return pack_contrast(target - control, seed_indices, day_samples[target.shape[1]])

        nominal.append(panel(label, 11))
        ood.append(panel(label, 7))
        for key, name in METHODS.items():
            m = suites["confirm_16d"]["methods"][key]
            a = m["mean"]
            effect = "--" if key == REF else interval(contrast(key, "confirm_16d"))
            nominal.append(row(name, error_bar(a[J], m["seed_std"][J]),
                               *(f"{a[k]:.2f}" for k in (WAIT, ONBOARD)),
                               f"{a[CV]:.4f}", f"{a['plan_cv2']:.4f}",
                               *(f"{a[k]:.2f}" for k in (EWT, READY, "hold_s_per_trip")),
                               f"{100 * a['passenger_unserved_rate']:.3f}", effect))
            regimes = ("ood_peak_shift", "ood_hf_surge", "ood_congestion")
            cols = [suites[s]["methods"][key]["mean"] for s in regimes]
            ood.append(row(name, *(f"{m[k]:.2f}" for m in cols for k in (J, READY))))
            for s in regimes:
                metric(key, s)
            if "null_flat_sanity" in suites:
                suite = "null_flat_sanity"
                a = suites[suite]["methods"][key]["mean"]
                effect = "--" if key == REF else interval(contrast(key, suite))
                transfer.append(row(name, f"{a[J]:.2f}", f"{a[READY]:.2f}", effect))
    rows("nominal", nominal)
    rows("ood", ood)
    rows("transfer", transfer)


def development_tables():
    v500 = read(EXPERIMENTS / "v500_pathb_headroom_gate/review03/aggregate/results.json")
    v501 = read(EXPERIMENTS / "v501_pathb_b1_stability/review03/aggregate/results.json")
    probe, checkpoints = [], []
    for c, name in (*CORRIDORS.items(), ("flat_sanity", "Flat-Null")):
        b = v500["by_corridor"][c]
        exps = b["paired_contrasts"]
        show = [(r"$A_3-A_2$", "A3_minus_A2_unbudgeted_leakage",
                 "A3_lf_quota_unbudgeted_hold"),
                (r"$A_4-A_3$", "A4_minus_A3_budget_repair",
                 "A4_lf_quota_turnaround_hf_hold")]
        probe.append(panel(name, 5))
        for label, key, arm in show:
            e = exps[key]
            probe.append(row(label, interval((e[J]["mean"], *e[J]["ci95"])),
                             f"{e[EWT]['mean']:+.2f}", f"{b['arm_means'][arm][READY]:.2f}",
                             f"{b['arm_means'][arm]['hold_s_per_trip']:.2f}"))
        if c in CORRIDORS:
            b = v501["by_corridor"][c]
            checkpoints.append(panel(name, 5))
            for ep, a in b["b1_milestones"].items():
                m, s = a["across_seed_mean"], a["across_seed_std"]
                checkpoints.append(row(ep.removeprefix("ep"), error_bar(m[J], s[J]),
                                       *(f"{m[k]:.2f}" for k in (WAIT, EWT, READY))))
    rows("probe", probe)
    rows("checkpoints", checkpoints)


if __name__ == "__main__":
    TABLES.mkdir(exist_ok=True)
    v502_tables()
    development_tables()
