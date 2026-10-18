# Build the working paper: run `make` from this folder.
# The compiled PDF is copied to fx-portfolio-lab.pdf, the versioned copy linked from the README.
export PATH := /Library/TeX/texbin:$(PATH)
LATEX = pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build

fx-portfolio-lab.pdf: build/main.pdf
	cp build/main.pdf fx-portfolio-lab.pdf

build/main.pdf: main.tex macros.tex references.bib sections/*.tex tables/*.tex figures/*.png
	mkdir -p build
	$(LATEX) main.tex > /dev/null
	BIBINPUTS=. bibtex build/main > /dev/null
	$(LATEX) main.tex > /dev/null
	$(LATEX) main.tex | grep -E "Output written|Warning|Error" || true

clean:
	rm -rf build

.PHONY: clean
