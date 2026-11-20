#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
paper=$PWD
root=$(cd ../.. && pwd)
test -f "$root/experiments/v505_holding_attribution/confirmation/aggregate.json"
bash build.sh
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/submission/generated" "$stage/review"

latexpand main.tex > "$stage/submission/main.tex"
cp sn-jnl.cls sn-basic.bst refs.bib build/main.pdf "$stage/submission/"
cp generated/circulation.pdf "$stage/submission/generated/"
(
    cd "$stage/submission"
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$stage/check" main.tex
    if rg -ni 'warning.*(undefined|multiply defined)|there were undefined' "$stage/check/main.log"; then
        exit 1
    fi
    zip -q "$paper/build/FreqDuet_Public_Transport_source.zip" \
        main.tex main.pdf sn-jnl.cls sn-basic.bst refs.bib generated/circulation.pdf
)

cd "$root"
pt=paper/public_transport_freqduet_v7_202611
v505=experiments/v505_holding_attribution
v504=experiments/v504_corrected_comparators
v503=experiments/v503_matched_routing
v502=experiments/v502_pathb_trc_matrix
v501=experiments/v501_pathb_b1_stability
v500=experiments/v500_pathb_headroom_gate
tar -cf - \
    "$pt"/{README.md,build.sh,main.tex,introduction.tex,methods.tex,results.tex,discussion.tex,implementation.tex,refs.bib,sn-jnl.cls,sn-basic.bst,render_evidence.py,rebuild_mbta_input.py} \
    "$v505"/{analyze.py,runner.py,field_runtime.py,vehicle_runtime.py,vehicle_crosswalk.py,PROTOCOL.md,RESULTS.md,confirmation} \
    "$v504"/{study.py,plan.json,matched_controls.py,runtime.patch,PROTOCOL.md,confirmation} \
    "$v503"/{study.py,plan.json,runtime.patch,PROTOCOL.md,submission_train.json,confirmation} \
    "$v502"/{PROTOCOL.md,trc_matrix_runner.py,review/aggregate/results.json} \
    "$v501"/freqduet_v7_b1.py \
    "$v500"/{build_corridors.py,corridors,review03/aggregate/results.json} \
    freqduet/env/config.json freqduet/env/data/time_table.xlsx \
    | tar -xf - -C "$stage/review"
bash "$stage/review/$pt/build.sh" "$stage/review_build"
cp "$stage/review_build/main.pdf" "$stage/review/$pt/main.pdf"
tar -czf "$paper/build/FreqDuet_Public_Transport_reproducibility.tar.gz" -C "$stage/review" .
