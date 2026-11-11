# V407 Engineering Acceptance

Task t214721 passed all six synthetic groups on node004 with zero real traffic:
seven native-action continuations match V406 where the service scopes coincide;
later published trips use the original fleet; unfinished vehicles, passenger
stock and overdue dispatch obligations remain accounted at the window boundary;
known demand/end clocks, stock balance and native legal masks pass.

Fixed 12-vehicle/20-stop fixture, 14 evaluations: frozen kernel 0.2875 s / 21,140
events; continuous-service kernel 0.6325 s / 45,444 events. More service is modeled,
so this is not an equivalent-output speed benchmark or a full-day runtime bound.

Two earlier zero-traffic failures are retained: t214716 asserted every added trip
must already launch instead of allowing pending obligations; t214720 gave the
cap fixture an illegal native request outside its mask. Both repairs affect test
fixtures only, not the controller or kernel. source01/source02 remain untouched;
source03/run03 is the accepted server snapshot. All 16 traffic slots remain
available after engineering. Evidence: review/checks03/results.json.

The bounded four-day/four-arm development protocol is sealed in protocol.json.
Only scheduler high/manual CPU-node jobs are used; no training or promotion.
