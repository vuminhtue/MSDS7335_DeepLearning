# MSDS7335 Final Report - LaTeX Documentation

This folder contains the comprehensive LaTeX report for the MSDS7335 Machine Learning II final project on automated RAG chatbot development using Cursor.com.

## 📁 Files Overview

### Main Documents
- **`MSDS7335_Final_Report.tex`** - Main LaTeX source file for the report
- **`references.bib`** - Bibliography file with APA-style references
- **`MSDS7335_Presentation.pptx`** - Original presentation file (reference material)

### Compilation Tools
- **`compile_report.sh`** - Automated compilation script for the LaTeX document
- **`README_Report.md`** - This documentation file

## 🔧 Prerequisites

Before compiling the report, ensure you have the following installed:

### Required Software
1. **LaTeX Distribution** (one of the following):
   - **TeX Live** (recommended for Linux/macOS): https://www.tug.org/texlive/
   - **MiKTeX** (recommended for Windows): https://miktex.org/
   - **MacTeX** (macOS): https://www.tug.org/mactex/

2. **Required LaTeX Packages** (usually included in full distributions):
   - `times` - Times New Roman font
   - `natbib` - Natural bibliography
   - `apacite` - APA citation style
   - `listings` - Code listing support
   - `xcolor` - Color support
   - `graphicx` - Image support
   - `hyperref` - Hyperlink support
   - `fancyhdr` - Header/footer formatting
   - `booktabs` - Professional table formatting

## 🚀 Compilation Instructions

### Method 1: Using the Automated Script (Recommended)

```bash
# Navigate to the report folder
cd report

# Run the compilation script
./compile_report.sh
```

The script will:
1. Check for required LaTeX tools
2. Perform multiple compilation passes
3. Process the bibliography
4. Generate the final PDF
5. Optionally clean up auxiliary files

### Method 2: Manual Compilation

If you prefer manual control or need to troubleshoot:

```bash
# Navigate to the report folder
cd report

# First compilation pass
pdflatex MSDS7335_Final_Report.tex

# Process bibliography
bibtex MSDS7335_Final_Report

# Second compilation pass (resolves citations)
pdflatex MSDS7335_Final_Report.tex

# Final compilation pass (resolves cross-references)
pdflatex MSDS7335_Final_Report.tex
```

### Method 3: Using LaTeX Editors

Popular LaTeX editors with built-in compilation:
- **Overleaf** (online): Upload all files to a new project
- **TeXShop** (macOS): Open the .tex file and press "Typeset"
- **TeXworks** (cross-platform): Open and compile
- **VS Code** with LaTeX Workshop extension

## 📄 Report Structure

The report follows a 4-chapter academic structure:

### Chapter 1: Introduction
- Background and motivation for the project
- Role of AI-assisted development
- Research objectives and contributions
- Report structure overview

### Chapter 2: Models and Data
- Technical architecture overview
- Ollama local LLM infrastructure
- Langchain RAG framework
- ChromaDB vector database management
- Gradio user interface development
- Complete RAG workflow implementation

### Chapter 3: Automated Workflow via Cursor.com
- AI-assisted development methodology
- Project development process
- Key implementation achievements
- Deployment strategies (Local PC, HPC, AWS Cloud)
- Performance benchmarking

### Chapter 4: Discussion and Conclusion
- Project outcomes and achievements
- The necessity and impact of Cursor.com
- Deployment strategy validation
- Limitations and future work
- Broader implications for AI development

## 🎨 Document Features

### Professional Formatting
- 12pt Times New Roman font
- Double-spaced text
- 1-inch margins
- Professional header/footer
- Automatic page numbering

### Academic Standards
- APA citation style
- Comprehensive bibliography
- Table of contents
- List of figures and tables
- Professional code listings
- Proper cross-referencing

### Visual Elements
- Syntax-highlighted code snippets
- Professional tables with booktabs
- Figure placeholders for diagrams
- Consistent formatting throughout

## 🔍 Troubleshooting

### Common Issues and Solutions

**Issue**: `pdflatex: command not found`
- **Solution**: Install a LaTeX distribution (TeX Live, MiKTeX, or MacTeX)

**Issue**: Missing package errors
- **Solution**: Install the missing packages through your LaTeX distribution's package manager
- For TeX Live: `tlmgr install <package-name>`
- For MiKTeX: Use the MiKTeX Console

**Issue**: Bibliography not appearing
- **Solution**: Ensure you run the complete compilation sequence (pdflatex → bibtex → pdflatex → pdflatex)

**Issue**: Figures not displaying
- **Solution**: The report currently contains figure placeholders. Add actual image files to display them.

**Issue**: Long compilation time
- **Solution**: This is normal for the first compilation. Subsequent runs will be faster.

## 📊 Expected Output

After successful compilation, you should have:
- **`MSDS7335_Final_Report.pdf`** - The final formatted report (~15-20 pages)
- Clean, professional academic formatting
- Properly formatted bibliography
- Table of contents with page numbers
- All cross-references resolved

## 📝 Customization

### Adding Your Information
Edit the following sections in `MSDS7335_Final_Report.tex`:

```latex
\author{
    [Your Name] \\           % Replace with your actual name
    Southern Methodist University \\
    Lyle School of Engineering \\
    Master of Science in Data Science \\
    \vspace{0.25in}
}
```

### Adding Figures
To add actual figures, place image files in the report folder and update references:

```latex
\includegraphics[width=0.9\textwidth]{your_figure.png}
```

### Modifying Content
The LaTeX source is well-commented and structured for easy modification. Each chapter is clearly marked and can be edited independently.

## 🆘 Support

If you encounter issues:
1. Check the `.log` file for detailed error messages
2. Ensure all required packages are installed
3. Verify that your LaTeX distribution is up to date
4. Try compiling a simple test document to verify your LaTeX installation

## 📋 Notes

- The report is designed to meet academic standards for graduate-level coursework
- All citations follow APA style as commonly required in data science programs
- The document structure can be easily adapted for other similar projects
- The automated compilation script works on Unix-like systems (Linux, macOS) and Windows with appropriate shell support

---

**Report Generation Date**: December 2024  
**LaTeX Engine**: pdfLaTeX  
**Citation Style**: APA via apacite package  
**Total Expected Pages**: 15-20 pages