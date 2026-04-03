# 🔍 Multimodal Crime / Incident Report Analyzer

**AI for Engineers — Spring 2026 | Group Project 3**

An AI-powered system that ingests five types of unstructured crime data — 911 audio calls, police report PDFs, scene photographs, CCTV surveillance footage, and crime news text — and merges them into a **unified, structured incident report** with severity classification, LLM-generated summaries, and an interactive Streamlit dashboard.

---

## 👥 Team Members

| Member | Role | Module | Key AI Models |
|--------|------|--------|---------------|
| **Lokeshwar Reddy Peddarangareddy** | Audio Analyst + Integration Lead | `audio/`, `integration/` | OpenAI Whisper, spaCy NER, Flan-T5 |
| **Neha Reddy Poreddy** | Document Analyst | `pdf/` | PyMuPDF, pytesseract OCR, spaCy NER |
| **Nagraj V Vallakati** | Image Analyst | `images/` | YOLOv8 (fine-tuned + COCO), ViT, pytesseract |
| **Rahul Sarma Vogeti** | Video Analyst | `video/` | YOLOv8, OpenCV, CLIP ViT |
| **Swet Vimalkumar Patel** | Text Analyst | `text/` | Zero-shot classification, spaCy NER |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                         │
│  🎙️ 911 Calls  📄 Police PDFs  🖼️ Photos  🎬 CCTV  📝 News     │
└────────┬───────────┬───────────┬──────────┬──────────┬──────────┘
         │           │           │          │          │
    ┌────▼────┐ ┌────▼────┐ ┌───▼────┐ ┌──▼───┐ ┌───▼────┐
    │  Audio  │ │   PDF   │ │ Image  │ │Video │ │  Text  │
    │Pipeline │ │Pipeline │ │Pipeline│ │Pipe. │ │Pipeline│
    └────┬────┘ └────┬────┘ └───┬────┘ └──┬───┘ └───┬────┘
         │           │          │         │         │
    audio_output  pdf_output  image_out  video_out  text_output
      .csv          .csv       .csv       .csv       .csv
         │           │          │         │         │
         └───────────┴──────┬───┴─────────┴─────────┘
                            │
                   ┌────────▼────────┐
                   │   Integration   │
                   │  Sequential     │
                   │  Merge Pipeline │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  LLM Summarizer │
                   │  (Flan-T5-base) │
                   └────────┬────────┘
                            │
                  final_incidents.csv
                   (198 incidents)
                            │
                   ┌────────▼────────┐
                   │    Streamlit    │
                   │    Dashboard    │
                   └─────────────────┘
```

---

## 📊 Pipeline Output Schemas

Each pipeline produces a structured CSV following the assignment specification:

| Pipeline | Output Columns | Records |
|----------|---------------|---------|
| **Audio** | `Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score` | 703 |
| **PDF** | `Report_ID, Department, Incident_Type, Doc_Type, Date, Location, Program, Officer, Summary, Key_Detail` | 45 |
| **Image** | `Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Text_Extracted, Confidence_Score` | 100 |
| **Video** | `Clip_ID, Timestamp, Frame_ID, Event_Detected, Persons_Count, Objects, Confidence` | 198 |
| **Text** | `Text_ID, Source, Raw_Text, Crime_Type, Location_Entity, Sentiment, Entities, Topic, Severity_Label` | 113 |

### Integrated Output

`final_incidents.csv` — 198 unified incidents with columns:
`Incident_ID, Audio_Event, PDF_Doc_Type, Image_Objects, Video_Event, Text_Crime_Type, Severity, Sources_Available, Modality_Count, AI_Summary`

---

## 🚀 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Lokeshloki1219/Multimodal-Incident-Analyzer.git
cd multimodal-incident-analyzer

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

# Install all dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## ▶️ Running Each Pipeline

```bash
# Student 1 — Audio Pipeline (Lokeshwar)
python audio/audio_pipeline.py

# Student 2 — PDF Pipeline (Neha)
python pdf/pdf_pipeline.py

# Student 3 — Image Pipeline (Nagraj)
python images/image_pipeline.py

# Student 4 — Video Pipeline (Rahul)
python video/video_pipeline.py

# Student 5 — Text Pipeline (Swet)
python text/text_pipeline.py
```

---

## 🔗 Running Integration + Dashboard

```bash
# Step 1: Merge all 5 modality CSVs into final_incidents.csv
python integration/integrate.py

# Step 2 (Optional): Generate AI summaries using Flan-T5
python integration/summarizer.py

# Step 3: Launch the Streamlit dashboard
streamlit run integration/dashboard.py
```

The dashboard opens at `http://localhost:8501` with:
- Severity distribution and modality coverage charts
- Audio event and text crime type breakdowns
- Per-incident detail view with AI-generated summaries
- Per-modality data explorer tabs

---

## ⚡ Real-Time Processing

The system supports real-time file ingestion using Python's `watchdog` library:

```bash
# Terminal 1 — Start the file monitor
python integration/realtime_monitor.py

# Terminal 2 — Start the dashboard
streamlit run integration/dashboard.py
```

Then **drag and drop** a file into the `new_data/` folder:

| File Type | Extension | Routed To |
|-----------|-----------|-----------|
| Audio | `.wav`, `.mp3`, `.flac` | Audio Pipeline |
| PDF | `.pdf` | PDF Pipeline |
| Image | `.jpg`, `.png`, `.bmp` | Image Pipeline |
| Video | `.mp4`, `.avi`, `.mpg` | Video Pipeline |
| Text | `.csv`, `.txt`, `.json` | Text Pipeline |

The monitor automatically:
1. Detects the file type from its extension
2. Runs the appropriate pipeline
3. Appends the result to the modality CSV
4. Creates a new incident in `final_incidents.csv`
5. Generates an AI summary (Flan-T5) for the new row only
6. The dashboard auto-refreshes every 10 seconds

---

## 📁 Project Structure

```
multimodal-incident-analyzer/
├── README.md                          # This file
├── requirements.txt                   # Global dependencies
├── .gitignore
│
├── audio/                             # Student 1 — Lokeshwar
│   ├── audio_pipeline.py              # Whisper + spaCy + sentiment
│   ├── download_dataset.py            # Dataset downloader
│   ├── audio_output.csv               # 703 transcribed 911 calls
│   ├── requirements.txt
│   └── data/                          # Audio WAV files (not tracked)
│
├── pdf/                               # Student 2 — Neha
│   ├── pdf_pipeline.py                # PyMuPDF + OCR + NER
│   ├── pdf_output.csv                 # 45 police report sections
│   ├── requirements.txt
│   └── data/                          # PDF files (not tracked)
│
├── images/                            # Student 3 — Nagraj
│   ├── image_pipeline.py              # Dual YOLOv8 + ViT + OCR
│   ├── train_fire_model.py            # Fire/smoke model trainer
│   ├── check_labels.py                # Label diagnostic utility
│   ├── image_output.csv               # 100 analyzed images
│   ├── requirements.txt
│   └── data/                          # Roboflow dataset (not tracked)
│
├── video/                             # Student 4 — Rahul
│   ├── video_pipeline.py              # OpenCV + YOLOv8 + CLIP ViT
│   ├── video_output.csv               # 198 key frame analyses
│   ├── requirements.txt
│   └── data/                          # CAVIAR CCTV clips (not tracked)
│
├── text/                              # Student 5 — Swet
│   ├── text_pipeline.py               # Zero-shot + spaCy + sentiment
│   ├── text_output.csv                # 113 crime news articles
│   ├── requirements.txt
│   └── data/                          # CrimeReport JSON (not tracked)
│
├── integration/                       # Integration + Dashboard
│   ├── integrate.py                   # Sequential merge pipeline
│   ├── summarizer.py                  # Flan-T5 LLM summarizer
│   ├── dashboard.py                   # Streamlit interactive dashboard
│   ├── realtime_monitor.py            # Watchdog file monitor
│   ├── requirements.txt
│   ├── final_incidents.csv            # 198 unified incidents
│   ├── final_detailed_dataset.csv     # Detailed export
│   ├── final_integrated_dataset.csv   # Integrated export
│   └── final_summary_dataset.csv      # Summary export
│
└── new_data/                          # Drop files here for real-time processing
    └── .gitkeep
```

---

## 🧠 AI Models Used

| Model | Purpose | Where |
|-------|---------|-------|
| **OpenAI Whisper** (base) | Speech-to-text transcription | Audio pipeline |
| **spaCy** (en_core_web_sm) | Named Entity Recognition | Audio, PDF, Text |
| **YOLOv8n** (COCO) | Object detection (80 classes) | Image, Video |
| **YOLOv8n** (fine-tuned) | Fire/smoke/human detection | Image |
| **ViT** (google/vit-base-patch16-224) | Scene classification | Image |
| **CLIP ViT** (openai/clip-vit-base-patch32) | Zero-shot scene classification | Video |
| **Zero-shot classifier** (facebook/bart-large-mnli) | Crime topic classification | Text |
| **Flan-T5-base** (google/flan-t5-base) | Incident summarization | Integration |
| **pytesseract** | OCR text extraction | PDF, Image |
| **HuggingFace Transformers** | Sentiment analysis | Audio, Text |

---

## 📈 Integration Strategy

The merge uses **sequential alignment** — since the five datasets come from independent sources with different row counts (Audio: 703, PDF: 45, Image: 100, Video: 198, Text: 113), incidents are assigned `INC_001` through `INC_198` with varying modality completeness:

| Range | Modalities | Count |
|-------|-----------|-------|
| INC_001 – INC_045 | Audio + PDF + Image + Video + Text | 45 |
| INC_046 – INC_100 | Audio + Image + Video + Text | 55 |
| INC_101 – INC_113 | Audio + Video + Text | 13 |
| INC_114 – INC_198 | Audio + Video | 85 |

Missing modality data is filled with context-aware descriptions (e.g., "No police report filed") rather than NaN values.

**Severity** is computed adaptively from available signals: Audio Urgency Score, Image Confidence Score, and Text Severity Label → averaged → High (>0.7) / Medium (>0.4) / Low (≤0.4).

---

## 📜 License

This project was developed as part of the AI for Engineers course at the University of Houston, Spring 2026.