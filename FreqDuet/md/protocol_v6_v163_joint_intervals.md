# V163: calibration-only joint maps

Task `t208318`, node004, high/manual, source `35ae15b185`: eight tests passed.
Use V162's 32 calibration days only, with V161 equal day/episode/query masses,
exact observed-CDF day ranks and same-query R/S/residence vectors. No pooling.
Sealed 10,562 candidate keys (4,861 finite) and 48 coarse reference keys;
3,104,478 finite queries from 3,134,878 native-supported joint atoms.
Support is episode-mean then actor/day-mean, not the pooled query fraction.
Old/handoff percentages: all 92.921/92.926; active 93.257/93.262;
clearance 80.408/80.426; hour2 91.927/91.900; hour13 79.400/79.465.
Every scope has the full 384-cell grid, finite estimates in every cell and no
missing reference. Both hour13 scopes fail the frozen 80% support threshold;
19,109/20,772 hour13 abstentions are keys below 19 distinct calibration days.
Maps are sealed but confirmation release is blocked. No confirmation, replay,
training or changes to V151/V146/window/tails/main policy. Four local summaries
total 31,337 bytes; maps, atoms and bounds remain on the server.
Next: read-only hour13 sparse-key decomposition by frozen conditioning fields;
do not regroup, refit or open confirmation to rescue this calibration result.
