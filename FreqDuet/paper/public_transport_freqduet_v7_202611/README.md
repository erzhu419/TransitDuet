# Public Transport revision

This is the fixed-fleet FreqDuet v7 manuscript for Public Transport, with
V504 as the main comparison and V505 as the submission revision's holding
intervention, rolling benchmark, and circulation case. The earlier TR-C
draft and the separate capped-holding study remain unchanged.

Run bash build.sh here to regenerate the compact tables and circulation
figure and compile build/main.pdf. Requirements: Python 3, NumPy,
Matplotlib, latexmk, and a TeX Live installation. The class and bibliography
style are unmodified files from the [official Springer Nature
template](https://www.springernature.com/gp/authors/campaigns/latex-author-support).

The frozen [V504 records](../../experiments/v504_corrected_comparators/confirmation/)
and the [V505 protocol](../../experiments/v505_holding_attribution/PROTOCOL.md)
identify the paired inputs. The review bundle includes the small numerical
archives and the published simulator inputs. It does not require downloaded
training weights, raw agency data, or an account on the scheduler.
The separate submission source bundle provides a flattened LaTeX file and
its figure and bibliography files.

MBTA's [Fall 2025 APC data](https://www.arcgis.com/sharing/rest/content/items/8daf4a33925a4df59183f860826d29ee/info/metadata/metadata.xml?format=default&output=html)
and [Spring 2026 GTFS stops](https://performancedata.mbta.com/) inform the
frozen input. Recreate the workbook with rebuild_mbta_input.py, supplying
explicit input paths. The October 2025 stop-event comparison uses a
separate [public archive](https://mbta-massdot.opendata.arcgis.com/datasets/924df13d845f4907bb6a6c3ed380d57a/about);
the selected turnaround gaps additionally use [Bus Observatory snapshots](https://api.busobservatory.org/mbta_all/schema).
Only compact route summaries are stored here.

An author must supply the title-page author list, affiliations, corresponding
email, funding and competing-interest disclosures, and confirm the AI-use
statement before submitting. No assertions about authors, funding, or
competing interests are inferred from other manuscripts.
