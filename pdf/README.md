# 📄 PDF Pipeline — Student 2 (Neha Reddy Poreddy)

## Overview

Extracts structured data from police report PDFs using digital text extraction with OCR fallback for scanned pages.

## Pipeline Stages

1. **Text Extraction** — PyMuPDF (fitz) extracts digital text from each PDF page
2. **OCR Fallback** — pytesseract OCR for scanned/image-based pages (rendered at 300 DPI)
3. **Section Splitting** — Detects document boundaries within multi-section PDFs
4. **Entity Extraction** — spaCy NER extracts persons, organizations, dates, locations
5. **Classification** — Rule-based incident type and document type classification
6. **Summary Generation** — Extractive summarization from document content

## Dataset

- **Source**: Police report PDFs (1033 Program training proposals, SOPs, inventory reports)
- **Format**: PDF files in `data/` subdirectory
- **Size**: Multi-page PDF documents → 45 document sections extracted

## Output

`pdf_output.csv` — 45 rows with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Report_ID` | String | Unique identifier (RPT_001, RPT_002, ...) |
| `Department` | String | Police department or agency name |
| `Incident_Type` | String | Classified incident type |
| `Doc_Type` | String | Document type (Training Proposal, SOP, Policy, etc.) |
| `Date` | String | Extracted date from document |
| `Location` | String | Extracted location |
| `Program` | String | Associated program name |
| `Officer` | String | Officer or author mentioned |
| `Summary` | String | Extractive summary of the document |
| `Key_Detail` | String | Primary key finding |

## Usage

```bash
python pdf/pdf_pipeline.py
```

## Dependencies

```
PyMuPDF (fitz)
pytesseract
spacy
pandas
Pillow
```
