# Multimodal Crime / Incident Report Analyzer

**AI for Engineers — Group Assignment**

## Team Members

| Member | Role | Module |
|--------|------|--------|
| Lokeshwar Reddy Peddarangareddy | Audio Analyst (Student 1) + Integration Lead | `/audio`, `/integration` |
| Neha Reddy Poreddy | Document Analyst (Student 2) | `/pdf` |
| Nagraj V Vallakati | Image Analyst (Student 3) | `/images` |
| Rahul Sarma Vogeti | Video Analyst (Student 4) | `/video` |
| Swet Vimalkumar Patel | Text Analyst (Student 5) | `/text`, `/dashboard` |

## Project Description

An AI-powered Multimodal Incident Analyzer that processes five types of unstructured data — audio calls, police PDFs, scene images, CCTV video, and crime text reports — and merges them into a single structured incident report with severity classification and an interactive dashboard.

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/team/multimodal-incident-analyzer.git
cd multimodal-incident-analyzer

# Install all dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Running Each Pipeline

```bash
# Audio Pipeline (Student 1 — Lokeshwar)
python audio/audio_pipeline.py

# PDF/Document Pipeline (Student 2 — Neha)
python pdf/pdf_pipeline.py

# Image Pipeline (Student 3 — Nagraj)
python images/image_pipeline.py

# Video Pipeline (Student 4 — Rahul)
python video/video_pipeline.py

# Text Pipeline (Student 5 — Swet)
python text/text_pipeline.py
```

## Running Integration

```bash
# Merge all 5 CSVs into final dataset
python integration/merge_pipeline.py

# Generate AI summaries (bonus)
python integration/llm_summarizer.py
```

## Launching the Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
multimodal-incident-analyzer/
├── README.md
├── requirements.txt
├── audio/
│   ├── audio_pipeline.py
│   ├── audio_output.csv
│   └── requirements.txt
├── pdf/
│   ├── pdf_pipeline.py
│   ├── pdf_output.csv
│   └── requirements.txt
├── images/
│   ├── image_pipeline.py
│   ├── image_output.csv
│   └── requirements.txt
├── video/
│   ├── video_pipeline.py
│   ├── video_output.csv
│   └── requirements.txt
├── text/
│   ├── text_pipeline.py
│   ├── text_output.csv
│   └── requirements.txt
├── integration/
│   ├── merge_pipeline.py
│   ├── llm_summarizer.py
│   ├── final_incidents.csv
│   └── incident_summary.csv
├── dashboard/
│   └── app.py
└── docs/
    ├── pipeline_architecture.png
    └── project_report.pdf
```
