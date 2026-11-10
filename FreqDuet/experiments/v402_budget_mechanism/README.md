# V402 budget-mechanism comparison

Three policies share the frozen manual-joint timetable and model fast layer:
turnover-pressure credit, static remaining-trip credit, and a stop cap. No actor
training or main-policy change. See PROTOCOL.md and protocol.json for the frozen
calibration grids, independent days, primary delay endpoint and CV2 protection.

`launch.py` stages and submits zero-traffic admission. `complete.py` orchestrates
engineering, calibration, sealed selection and independent confirmation, using
existing scheduler APIs and only small local summaries. All physical paths and
statistical calculations execute on node001--006, high/manual, at most eight
concurrent paths; failed starts are charged, not automatically retried.

Synthetic task t214090 passed six contracts with zero native Runner constructions.
Freshness task t214091 passed; engineering gate t214103 found zero differences
for both pressure reproduction and the stop-cap noop. Calibration uses eight
existing queues t214104--t214111. Registration recovery reused the first four
IDs without restarting any path; later stages are submitted in one batch.
Sources and raw results remain under the dedicated V402 server root; historical
TransitDuet/FreqDuet code and results are unchanged.
