"""Link stopped AVL departures to agency timepoints when the trip ID namespaces differ."""

from collections import Counter, defaultdict
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import duckdb
from field_runtime import clock, quantiles
from vehicle_runtime import COLUMNS, EASTERN, FILES

ORIGIN = {1: "5547", 0: "8309"}
TERMINAL = {1: "8309", 0: "55471"}


def agency(archive):
    trips = defaultdict(dict)
    duplicates = set()
    with ZipFile(archive) as z:
        name, = [n for n in z.namelist() if n.endswith("2025-10.csv")]
        with z.open(name) as f:
            rows = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
            for r in rows:
                day = r["service_date"]
                if r["route_id"] != "111" or not ("2025-10-06" <= day <= "2025-10-10"):
                    continue
                d = {"Inbound": 1, "Outbound": 0}[r["direction_id"]]
                t = clock(r["actual"])
                if not r["half_trip_id"] or t is None:
                    continue
                trip = day, d, r["half_trip_id"]
                stop = r["stop_id"]
                if stop in trips[trip] and trips[trip][stop] != t:
                    duplicates.add(trip)
                trips[trip][stop] = t
    return {k: x for k, x in trips.items()
            if k not in duplicates and ORIGIN[k[1]] in x and TERMINAL[k[1]] in x
            and 6 * 3600 <= x[ORIGIN[k[1]]] < x[TERMINAL[k[1]]] < 19 * 3600}


def vehicle_rows(paths):
    cols = [f'"{s}"' for s in COLUMNS]
    sql = f"""
    WITH rows AS (
      SELECT {cols[0]} AS bus,{cols[1]} AS trip,{cols[2]} AS route,{cols[3]} AS dir,
             CAST(epoch({cols[4]}) AS BIGINT) AS stamp,{cols[5]} AS stop,{cols[6]} AS status
      FROM read_parquet(?)
      WHERE {cols[0]} IS NOT NULL AND {cols[1]} IS NOT NULL AND {cols[4]} IS NOT NULL
    ), selected AS (SELECT DISTINCT bus FROM rows WHERE route='111')
    SELECT DISTINCT rows.* FROM rows JOIN selected USING(bus) ORDER BY bus,stamp
    """
    c = duckdb.connect(config={"threads": 1})
    days = defaultdict(dict)
    for b, trip, route, direction, stamp, stop, status in c.execute(sql, [paths]).fetchall():
        dt = datetime.fromtimestamp(stamp, EASTERN)
        if not (6 <= dt.day <= 10 and 6 <= dt.hour < 19):
            continue
        a = days[b, dt.date().isoformat()].setdefault((trip, route, direction), [stamp, stamp, {}])
        a[1] = stamp
        if route == "111" and status == 1 and stop is not None:
            a[2][stop] = dt.hour * 3600 + dt.minute * 60 + dt.second
    return days


def measure(archive, paths):
    published = agency(archive)
    obs = vehicle_rows(paths)
    indexed = defaultdict(list)
    for k in published:
        indexed[k[:2]].append(k)
    candidates = {}
    reverse = Counter()
    for (bus, day), trips in obs.items():
        for trip, (_, _, stops) in trips.items():
            _, route, d = trip
            if route != "111" or d not in ORIGIN or ORIGIN[d] not in stops:
                continue
            k = (bus, day, trip)
            for a in indexed[day, d]:
                observed = published[a]
                if abs(stops[ORIGIN[d]] - observed[ORIGIN[d]]) > 90:
                    continue
                if not any(abs(stops[s] - observed[s]) <= 90
                           for s in stops.keys() & observed.keys() - {ORIGIN[d], TERMINAL[d]}):
                    continue
                candidates.setdefault(k, []).append(a)
                reverse[a] += 1
    matched = {k: x[0] for k, x in candidates.items() if len(x) == 1 and reverse[x[0]] == 1}
    turnaround = defaultdict(list)
    days = defaultdict(set)
    for (bus, day), trips in obs.items():
        ordered = sorted(trips, key=lambda k: trips[k][0])
        for p, q in zip(ordered, ordered[1:]):
            a, b = matched.get((bus, day, p)), matched.get((bus, day, q))
            if a is None or b is None or a[1] == b[1] or trips[p][1] > trips[q][0]:
                continue
            elapsed = published[b][ORIGIN[b[1]]] - published[a][TERMINAL[a[1]]]
            if 0 < elapsed <= 3600:
                key = "Haymarket" if b[1] == 0 else "Woodlawn"
                turnaround[key].append(elapsed)
                days[key].add(day)
    return {
        "primary_snapshot_endpoint_check": "No destination STOPPED_AT events at the modal terminal IDs",
        "method": "Unique reciprocal origin plus interior departure match, each within 90 seconds",
        "vehicle_source": "Cornell Tech Bus Observatory (CC BY-NC 4.0)",
        "agency_source": "MBTA Bus Arrival Departure Times 2025",
        "files": FILES, "usable_agency_trips": len(published),
        "vehicle_day_groups": len(obs), "uniquely_linked_trips": len(matched),
        "turnaround": {k: {"n": len(v), "days": len(days[k]), "seconds": quantiles(v)}
                       for k, v in sorted(turnaround.items())}}


if __name__ == "__main__":
    folder, archive, output = map(Path, sys.argv[1:])
    out = measure(archive, [str(folder / n) for n in FILES])
    target = output / "summary"
    target.mkdir(parents=True, exist_ok=False)
    (target / "results.json").write_text(json.dumps(out, indent=2) + "\n")
