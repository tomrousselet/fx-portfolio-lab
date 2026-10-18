# Build the working paper: run `make` from this folder; the PDF is build/main.pdf
export PATH := /Library/TeX/texbin:$(PATH)
LATEX = pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build

build/main.pdf: main.tex macros.tex references.bib sections/*.tex
	mkdir -p build
	$(LATEX) main.tex > /dev/null
	BIBINPUTS=. bibtex build/main > /dev/null
	$(LATEX) main.tex > /dev/null
	$(LATEX) main.tex | grep -E "Output written|Warning|Error" || true

clean:
	rm -rf build

.PHONY: clean
