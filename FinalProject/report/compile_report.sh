#!/bin/bash

# LaTeX Report Compilation Script
# This script compiles the MSDS7335 Final Report LaTeX document

echo "🔧 Compiling MSDS7335 Final Report..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)."
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "❌ Error: bibtex not found. Please install a LaTeX distribution with BibTeX support."
    exit 1
fi

# Set the main document name
DOCUMENT="MSDS7335_Final_Report"

echo "📝 First LaTeX compilation..."
pdflatex -interaction=nonstopmode "$DOCUMENT.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error during first LaTeX compilation. Check the .log file for details."
    exit 1
fi

echo "📚 Processing bibliography..."
bibtex "$DOCUMENT"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: BibTeX processing had issues. Continuing..."
fi

echo "📝 Second LaTeX compilation..."
pdflatex -interaction=nonstopmode "$DOCUMENT.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error during second LaTeX compilation. Check the .log file for details."
    exit 1
fi

echo "📝 Final LaTeX compilation..."
pdflatex -interaction=nonstopmode "$DOCUMENT.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error during final LaTeX compilation. Check the .log file for details."
    exit 1
fi

echo "✅ Report compilation completed successfully!"
echo "📄 Output file: $DOCUMENT.pdf"

# Clean up auxiliary files (optional)
read -p "🗑️  Clean up auxiliary files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.aux *.bbl *.blg *.log *.lof *.lot *.toc *.out *.fdb_latexmk *.fls *.synctex.gz
    echo "🧹 Auxiliary files cleaned up."
fi

echo "🎉 Done! Your report is ready: $DOCUMENT.pdf"