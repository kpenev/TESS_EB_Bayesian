all: proposal

proposal:
	-doc2pdf WorkEffortTable_anonymized.docx
	-doc2pdf budget/budget_justification.docx
	-doc2pdf budget/budget_justification_anonymized.docx
	-doc2pdf WorkEffortTable.docx
	-pdflatex --shell-escape proposal.tex
	-bibtex proposal
	-pdflatex --shell-escape proposal.tex
	-pdflatex --shell-escape proposal.tex
	-pdflatex --shell-escape anonymized.tex
	-pdflatex --shell-escape non_anonymized.tex
#	-pdflatex --shell-escape summary_pages.tex
#	-pdflatex --shell-escape project_description_pages.tex
#	-pdflatex --shell-escape references.tex
#	-pdflatex --shell-escape data_management_plan.tex
#	-pdflatex --shell-escape facilities_and_equipment_pages.tex
#	-pdflatex --shell-escape suggested_reviewers.tex

clean:
	rm *.out *.aux *.toc *.bbl *.blg *.log *.dvi proposal.pdf
