# PDF Pipeline - Student 2 (Neha Reddy Poreddy)
## Multimodal Crime / Incident Report Analyzer

### What's Included
- pdf_pipeline.py - Main pipeline script
- 
equirements.txt - Python dependencies
- pdf_output.csv - Pre-generated output (5 records from LESO2.pdf)
- data/LESO2.pdf - MuckRock Arkansas Police PDF (75 pages)

### Setup Steps (Run on Your System)

1. **Install Python 3.10+** (if not already installed)

2. **Create a virtual environment:**
   `ash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   `

3. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   `

4. **Run the pipeline:**
   `ash
   python pdf_pipeline.py
   `

5. **Output:** pdf_output.csv with columns:
   Report_ID, Incident_Type, Date, Location, Officer, Summary

### AI Tools Used
- **PyMuPDF (fitz)** - PDF text extraction
- **pdfplumber** - Table extraction from PDFs
- **pytesseract** - OCR for scanned PDFs (needs Tesseract installed)
- **spaCy** - Named Entity Recognition (dates, names, locations)
- **BART (HuggingFace)** - Text summarization

### Dataset
- **Source:** MuckRock FOIA - Arkansas Police Department 1033 Training Plan
- **Link:** https://www.muckrock.com
- **Already included** in data/LESO2.pdf

### To Upload to GitHub
1. Place entire pdf/ folder in repo root
2. Commit and push
