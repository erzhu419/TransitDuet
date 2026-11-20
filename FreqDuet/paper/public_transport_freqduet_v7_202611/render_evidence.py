"""Rebuild the manuscript tables and the circulation figure from archived data."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
STUDY = EXP / "v505_holding_attribution"
sys.path.insert(0, str(STUDY))
from analyze import PRIMARY, table

OUT = HERE / "generated"
J, CV, READY, PUBLIC = PRIMARY
LABEL = {
    J: "Journey (s/rider)", "restricted_wait_s_per_pax": "Waiting (s/rider)",
    "restricted_in_vehicle_s_per_pax": "Onboard (s/rider)", CV: r"Arrival CV$^2$",
    READY: "Readiness (s/trip)", PUBLIC: "Abs. offset (s/trip)",
    "hold_s_per_trip": "Holding (s/trip)",
}
KEYS = (J, "restricted_wait_s_per_pax", "restricted_in_vehicle_s_per_pax", CV, READY,
        PUBLIC, "hold_s_per_trip")


def put(name, text):
    (OUT / (name + ".tex")).write_text(text.strip() + "\n")


def interval(e, digits=2):
    m = f"{e['mean']:+.{digits}f}"
    a, b = (f"{x:+.{digits}f}" for x in e["ci95"])
    return f"$ {m}\\; [{a},\\,{b}] $"


def contrast(columns, effects):
    def row(items):
        return " & ".join(items) + r" \\"
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l" + "r" * len(columns) + "@{}}", r"\toprule",
             row(("Outcome", *columns)), r"\midrule"]
    for key in KEYS:
        lines.append(row([LABEL[key], *(interval(e["metrics"][key], 5 if key == CV else 2)
                                         for e in effects)]))
    lines.append(row(["Joint criterion", *("Met" if e["joint_pass"] else "Not met" for e in effects)]))
    return "\n".join(lines + [r"\botrule", r"\end{tabular}"])


def hhmm(s):
    m = int(s) // 60 + 360
    return f"{m // 60:02d}:{m % 60:02d}"


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.3, 3.75))
    for i, (arm, name) in enumerate((("trace_aligned", "Aligned"), ("trace_nofreq", "NoFreq"))):
        p = STUDY / "confirmation" / f"{arm}_50314_0/results.json"
        trace = json.loads(p.read_text())["per_day"][0]["trace"]
        stock = np.asarray(trace["stock"], dtype=float)
        dispatch = {r[0]: r for r in trace["dispatch"]}
        first = min(trace["denials"], key=lambda r: r[2])
        last = max(trace["denials"], key=lambda r: dispatch[r[0]][3])
        for j, d in enumerate((1, 0)):
            ax = axes[i, j]
            ax.step(6 + stock[:, 0] / 3600, stock[:, 1 if d else 2], where="post",
                    lw=1.1, color="#256183")
            for key, direction, t, *_ in trace["denials"]:
                if direction != d:
                    continue
                row = dispatch[key]
                ax.hlines(-.45, 6 + t / 3600, 6 + row[3] / 3600, color="#ad382a", lw=1.75)
                if row[7] is not None and row[7] >= t:
                    ax.scatter(6 + row[7] / 3600, -.45, s=12, marker="|", color="#171717")
            for call, stamp, text in ((first, first[2], "First denial"),
                                      (last, dispatch[last[0]][3], "Last release")):
                if call[1] != d:
                    continue
                x = 6 + stamp / 3600
                ax.vlines(x, -.45, 6.15, colors="#ad382a", linestyles=":", lw=.85)
                align = ("right" if call == first else "left") if first[1] == last[1] else (
                    "left" if x < 14 else "right")
                ax.text(x, 6.25, f"{text}\n{hhmm(stamp)}", color="#803226", fontsize=8,
                        ha=align, va="bottom")
            ax.set_title(f"{name} | Terminal {'A' if d else 'B'}", loc="left", fontsize=10)
            ax.set_ylim(-.85, 8.5)
            ax.set_yticks([0, 2, 4, 6, 8])
            ax.set_xticks([6, 9, 12, 15, 18, 21])
            ax.grid(axis="y", alpha=.20)
            if j == 0:
                ax.set_ylabel("Ready buses")
            if i == 1:
                ax.set_xlabel("Simulated service hour")
    fig.tight_layout()
    fig.savefig(OUT / "circulation.pdf")
    plt.close(fig)


def holding(data):
    if "v505" not in data:
        put("abstract_extension", "")
        put("holding_results", "The closed-loop holding contrasts are not yet available.")
        put("holding_discussion", "")
        return
    a = data["v505"]
    put("v505_table", contrast(("Zero hold", "Rule hold"), [a[k] for k in ("zero", "rule")]))
    put("v505_rolling_table", contrast(("Rolling",), [a["rolling"]]))
    z, r, m = [a[k]["metrics"][J] for k in ("zero", "rule", "rolling")]
    wording = ("reduced" if z["ci95"][1] < 0 else
               "increased" if z["ci95"][0] > 0 else "did not establish a reduction in")
    put("abstract_extension",
        f"With the upper policy frozen, learned holding {wording} journey exposure relative to zero holding "
        f"({interval(z)} seconds per passenger).")
    rows = []
    for label, key in (("learned", "audit_aligned"), ("rule", "rule"), ("zero", "zero"), ("Rolling", "rolling")):
        c = data["v505_budget_counts"][key]
        rows.append(f"{label}: {100*c['mask_removes_bin']/c['queries']:.2f}\\% narrowed, "
                    f"{c['preferred_masked']:,}/{c['queries']:,} preferred actions blocked")
    put("holding_results",
        f"Aligned-minus-comparator journey contrasts were {interval(z)} against zero holding, "
        f"{interval(r)} against rule holding, and {interval(m)} against Rolling (seconds per passenger). "
        r"\begin{table}[!ht]\centering\small"
        r"\caption{Closed-loop holding attribution using the V504 day set. "
        r"All differences are Aligned minus the comparator; units and intervals match Table~\ref{tab:main}.}"
        r"\label{tab:holding}\input{generated/v505_table}\end{table}"
        "\nRelative to zero holding, learned holding slightly reduced estimated waiting but increased "
        "onboard time by 0.61 seconds per passenger (interval 0.11 to 1.14); the two effects did not "
        "produce a reliable journey reduction. Relative to rule holding with the same upper actor, "
        "learning cut onboard time but did not pass the journey criterion."
        "\nThe budget narrowed the legal action set even at events at which it did not change the "
        "preferred action. The same-state mask diagnostic gave " + "; ".join(rows)
        + ". The zero-holding arm cannot have its preferred action blocked."
        "\nRolling had lower average readiness delay under its revised plan (1.81 vs. 3.65 seconds "
        "per trip), but early displacement from the original timetable averaged 2268.04 seconds "
        "per trip, compared with 319.70 for Aligned. This is a consequence of the tested finite "
        "forecast under free retiming; it is not a published-timetable-constrained optimizer. "
        r"\begin{table}[!ht]\centering\small"
        r"\caption{Aligned minus the matched rolling-forecast comparator on the V504 day set. "
        r"Units and intervals match Table~\ref{tab:main}.}"
        r"\label{tab:rolling}\input{generated/v505_rolling_table}\end{table}")
    put("holding_discussion",
        "The lower-controller contrasts include the upper policy's reactions to changes in vehicle stock. "
        "They do not estimate the direct effect of holding on an identical future timetable or tell us "
        "how a dispatch policy retrained without holding would behave. Under the tested demand, "
        "zero holding is a credible simpler operating option. The small spacing signal from learned "
        "holding does not justify claiming that extra holding improved complete journeys.")


def case_results(data):
    cases = data.get("case", {})
    if len(cases) != 2:
        put("case_results", "Vehicle-level traces are not yet available.")
        return
    figure()
    lines = []
    for arm, label in (("trace_aligned", "Aligned"), ("trace_nofreq", "NoFreq")):
        v = cases[arm]
        first, end, d, trip = v["first_denial"]
        n = sum(v["denied_trips_by_terminal"].values())
        lines.append(f"{label} first lacked a ready bus at {hhmm(first)} in terminal {'A' if d else 'B'} "
                     f"(trip {trip}, released at {hhmm(end)}). {n} trips faced a supply denial; "
                     f"the last of those trips departed at {hhmm(v['last_recovery'])}.")
    put("case_results", "Both passive replays reproduced the archived endpoints exactly. " + " ".join(lines)
        + " In Aligned, 45 of 47 denied trips were at terminal B. For the first denial, the actual "
        "replacement vehicle returned from trip 24 one second before serving trip 39. The traces "
        "show a directional stock imbalance, but do not by themselves identify its causal components."
        + r" Figure~\ref{fig:case} links these delays to physical vehicle availability. "
        r"A tick on a red segment indicates that the vehicle assigned to that trip returned during the wait."
        "\n"
        r"\begin{figure}[!ht]\centering\includegraphics[width=\linewidth]{generated/circulation.pdf}"
        r"\caption{Selected circulation case, not a representative operating day. Blue: physical ready stock "
        r"sampled every minute. Red: from first no-bus denial to actual release; intervals may overlap. "
        r"Dotted lines mark the first denial and final release of a denied trip for each policy. "
        r"Hours are simulated local clock times.}\label{fig:case}\end{figure}")


def field_results(data):
    f = data.get("field")
    if f is None:
        put("field_results", "The independent stop-event comparison is not yet available.")
        return
    rows = []
    for direction, a in f["directions"].items():
        p10, p50, p90 = (x / 60.0 for x in a["runtime_s"])
        rows.append(f"{direction}: {a['trips']:,} observed trips across {a['weekdays']} weekdays, "
                    f"median {p50:.1f} minutes (10th--90th percentile {p10:.1f}--{p90:.1f})")
    case = data.get("case", {}).get("trace_aligned", {}).get("matched_previous_trip_times", {})
    sim = ""
    if len(case) == 2:
        x, y = (case[d]["trip_elapsed_s"][1] / 60 for d in ("1", "0"))
        sim = (f" The simulated terminal-to-terminal medians on the selected day "
               f"were {x:.1f} minutes inbound and {y:.1f} minutes outbound. "
               "The simulated outbound trip is shorter than observed. Endpoints and trip patterns differ, "
               "so the contrast is not a matched ground-truth error or a recalibration.")
    portal = "https://mbta-massdot.opendata.arcgis.com/datasets/924df13d845f4907bb6a6c3ed380d57a/about"
    statement = (
        rf"The \href{{{portal}}}{{MBTA bus stop-event archive}} independently records "
        "actual departures at timepoints. We retained complete weekday Route~111 half-trips starting "
        "from 06:00 to 19:00 in October 2025 and the modal endpoint pair per direction. "
        + "; ".join(rows) + ". All retained records use the agency's headway standard." + sim
        + " The agency stop-event fields do not identify the bus assigned to the next trip."
    )
    v = data.get("field_crosswalk")
    if v is not None:
        statement += (
            r" We also checked the \href{https://api.busobservatory.org/mbta_all/schema}"
            r"{Cornell Tech Bus Observatory} vehicle snapshots (Jacobs Urban Tech Hub, CC BY-NC 4.0) "
            "for 6--10 October 2025. The snapshots did not identify complete trips at both endpoints, "
            "and the two archives use different trip identifiers. We linked trips only when a unique "
            "one-to-one match agreed within 90 seconds at the origin and at an interior stop."
        )
        matched = []
        for term, a in v["turnaround"].items():
            p10, p50, p90 = (s / 60 for s in a["seconds"])
            matched.append(f"{term}: {a['n']} matched pairs across {a['days']} days, "
                           f"median {p50:.1f} minutes (10th--90th percentile {p10:.1f}--{p90:.1f})")
        if matched:
            statement += (
                f" This linked {v['uniquely_linked_trips']:,} of {v['usable_agency_trips']:,} usable agency trips. "
                "For consecutive opposite-direction trips observed on the same vehicle, agency "
                "end-to-start gaps of at most one hour were " + "; ".join(matched)
                + ". These are selected, time-matched connections, not a measure of vehicle "
                "or driver readiness or an agency-verified block assignment."
            )
            if len(case) == 2:
                a, b = (case[d]["realized_terminal_idle_s"][1] / 60 for d in ("1", "0"))
                statement += (
                    f" Across the selected simulated day, the same-vehicle medians after "
                    f"inbound and outbound trips were {a:.1f} and {b:.1f} minutes. The asymmetric "
                    "gaps describe that circulation failure, not typical operation of the route."
                )
        else:
            statement += " No consecutive opposite-direction pair met these requirements."
    else:
        statement += " These records alone cannot validate realized terminal turnaround."
    put("field_results", statement)


def main():
    OUT.mkdir(exist_ok=True)
    data = table()
    old = json.loads((EXP / "v504_corrected_comparators/confirmation/aggregate.json").read_text())["comparisons"]
    put("v504_table", contrast(("Manual", "NoFreq"),
                              [old["aligned_minus_manual"], old["aligned_minus_nofreq"]]))
    n = data["journey_savings_vs_manual_person_hours_per_day"]
    put("person_hours", f"{n:.1f}%")
    holding(data)
    case_results(data)
    field_results(data)


if __name__ == "__main__":
    main()
