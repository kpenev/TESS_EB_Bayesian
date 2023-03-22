all: proposal

proposal:
	-pdflatex --shell-escape $@.tex
	-bibtex $@
	-pdflatex --shell-escape $@.tex
	-pdflatex --shell-escape $@.tex
#	-pdflatex --shell-escape summary_pages.tex
#	-pdflatex --shell-escape project_description_pages.tex
#	-pdflatex --shell-escape references.tex
#	-pdflatex --shell-escape data_management_plan.tex
#	-pdflatex --shell-escape facilities_and_equipment_pages.tex
#	-pdflatex --shell-escape suggested_reviewers.tex

clean:
	rm *.out *.aux *.toc *.bbl *.blg *.log *.dvi proposal.pdf
