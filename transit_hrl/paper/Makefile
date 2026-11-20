PYTHON ?= python3
PANDOC ?= pandoc
BUNDLE ?= /tmp/freqhrl-submission.tar.gz
REVIEW_BUNDLE ?= /tmp/freqhrl-review.tar.gz
TMLR_SUPPLEMENT ?= /tmp/tmlr-supplement.zip

.PHONY: reproduce evidence figures check bundle review-bundle tmlr tmlr-package

submission.pdf: manuscript.md references.bib submission.lua Makefile $(wildcard figures/*.pdf)
	mkdir -p .submission
	$(PANDOC) manuscript.md --from=markdown+tex_math_single_backslash --standalone \
		--lua-filter=submission.lua --citeproc --bibliography=references.bib \
		-M reference-section-title=References \
		-V papersize=letter -V fontsize=11pt -V geometry:margin=0.72in \
		-V colorlinks=true -V linkcolor=black -V urlcolor=blue \
		-o .submission/article.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=.submission .submission/article.tex
	cp .submission/article.pdf $@

tmlr: tmlr/submission.pdf

tmlr/submission.pdf: manuscript.md references.bib submission.lua tmlr/template.tex \
		Makefile $(wildcard tmlr/upstream/*) $(wildcard figures/*.pdf)
	mkdir -p .submission/tmlr
	$(PANDOC) manuscript.md --from=markdown+tex_math_single_backslash --standalone \
		--lua-filter=submission.lua --natbib --bibliography=references.bib \
		--number-sections --syntax-highlighting=none -M tmlr=true \
		--template=tmlr/template.tex -o .submission/tmlr/tmlr-paper.tex
	TEXINPUTS="$(abspath tmlr/upstream):" BSTINPUTS="$(abspath tmlr/upstream):" \
		latexmk -Werror -pdf -interaction=nonstopmode -halt-on-error -outdir=.submission/tmlr .submission/tmlr/tmlr-paper.tex
	cp .submission/tmlr/tmlr-paper.pdf $@

evidence:
	$(PYTHON) build_plan_validity_evidence.py

figures: evidence
	$(PYTHON) render_plan_validity_figures.py

check:
	cd .. && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests $(PYTHON) -m unittest \
		test_paper_plan_validity_evidence test_plan_validity_figures

reproduce: figures
	$(MAKE) submission.pdf
	$(MAKE) check

bundle: reproduce
	cd .. && tar -czf "$(abspath $(BUNDLE))" -T paper/artifact_files.txt

review-bundle: reproduce
	$(PYTHON) build_review_bundle.py "$(abspath $(REVIEW_BUNDLE))"

tmlr-package: reproduce
	$(MAKE) tmlr
	$(PYTHON) build_review_bundle.py "$(abspath $(TMLR_SUPPLEMENT))"
