#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for tool in pandoc tectonic; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'missing required tool: %s\n' "$tool" >&2
        exit 2
    fi
done

pandoc manuscript_body.md \
    --from=markdown+citations+raw_tex \
    --to=latex \
    --natbib \
    --bibliography=references.bib \
    --template=elsarticle-template.tex \
    --metadata-file=metadata.yaml \
    --output=manuscript.tex

pandoc supplementary_body.md \
    --from=markdown+raw_tex \
    --to=latex \
    --template=supplementary-template.tex \
    --metadata-file=metadata.yaml \
    --output=supplementary.tex

tectonic --keep-logs --keep-intermediates manuscript.tex
tectonic --keep-logs --keep-intermediates supplementary.tex

test -s manuscript.pdf
test -s supplementary.pdf
if grep -E "Citation.*undefined|There were undefined references" manuscript.log; then
    printf 'unresolved references in manuscript.log\n' >&2
    exit 3
fi

printf 'TRC_BUILD_OK manuscript.pdf supplementary.pdf\n'
