.PHONY: all

all: sow_spin_only

%:
	-pdflatex --shell-escape $@.tex
	-bibtex $@
	-pdflatex --shell-escape $@.tex
	-pdflatex --shell-escape $@.tex

final: proposal
	-doc2pdf WorkEffortTable_anonymized.docx
	-doc2pdf budget/budget_justification.docx
	-doc2pdf budget/budget_justification_anonymized.docx
	-doc2pdf WorkEffortTable.docx
	-pdflatex --shell-escape anonymized.tex
	-pdflatex --shell-escape non_anonymized.tex
	pdf2ps anonymized.pdf
	ps2pdf -dPDFSETTINGS=/prepress anonymized.ps
	rm anonymized.ps
	pdf2ps non_anonymized.pdf
	ps2pdf -dPDFSETTINGS=/prepress non_anonymized.ps
	rm non_anonymized.ps


clean:
	rm *.out *.aux *.toc *.bbl *.blg *.log *.dvi proposal.pdf
