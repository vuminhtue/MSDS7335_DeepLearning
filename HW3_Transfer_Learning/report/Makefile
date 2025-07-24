# Makefile for Transfer Learning Report
# Author: MSDS 7335 - Machine Learning II

# Main document name
MAIN = transfer_learning_report_simple

# LaTeX compiler
LATEX = pdflatex

# BibTeX compiler  
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Compile PDF
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Clean temporary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Force rebuild
rebuild: cleanall all

# View PDF (macOS)
view: $(MAIN).pdf
	open $(MAIN).pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build the PDF report (default)"
	@echo "  clean     - Remove temporary LaTeX files"
	@echo "  cleanall  - Remove all generated files including PDF"
	@echo "  rebuild   - Clean everything and rebuild"
	@echo "  view      - Open the PDF (macOS)"
	@echo "  help      - Show this help message"

.PHONY: all clean cleanall rebuild view help 