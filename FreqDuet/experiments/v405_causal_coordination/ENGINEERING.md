# Recording Repair

The original 20 starts completed their native rollouts. Eighteen passed all
physical checks. Nominal 2450001/coordinated (t214537) and heavy
2450201/coordinated (t214547) reached all 5,200 decisions but failed in the
old budget auditor: a new dispatch-record kind was interpreted as a budget
return. Their failures remain charged; no automatic traffic retry occurred.

The fix uses the existing publication-record schema with an empty budget list.
No action, forecast, search, demand day or gate changes. Zero-traffic task
t214571 is not a traffic replacement; the authoritative pass is in
review/checks_recording/results.json.

Any authorized replacement must reproduce every saved native endpoint from
the original completed rollout before its full result is admitted. Failed
paths are excluded from paired summaries until then. The original matrices,
default policy and TransitDuet remain untouched.
