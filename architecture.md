# AI Pipeline Architecture — Multimodal Crime / Incident Report Analyzer

> **Course:** AI for Engineers | **Team Size:** 5 | **Due:** March 31, 2026 | **Total Marks:** 100

---

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph INGESTION["STAGE 1 — Unstructured Data Ingestion"]
        direction LR
        A1["911 Audio Calls<br/><i>Kaggle WAV files</i>"]
        A2["Police PDFs<br/><i>MuckRock LESO2</i>"]
        A3["Scene Photos<br/><i>Roboflow D-Fire</i>"]
        A4["CCTV Footage<br/><i>CAVIAR Clips</i>"]
        A5["Crime Text<br/><i>CrimeReport JSONL</i>"]
    end

    subgraph PROCESSING["STAGE 2 — AI Processing per Modality"]
        direction LR
        P1["Whisper STT<br/>spaCy NER<br/>HuggingFace Sentiment"]
        P2["PyMuPDF Extract<br/>spaCy NER<br/>BART Summarizer"]
        P3["YOLOv8 Detection<br/>ViT Classification<br/>Tesseract OCR"]
        P4["OpenCV Frames<br/>Motion Detection<br/>YOLOv8 Objects"]
        P5["NLTK Preprocess<br/>Zero-Shot Topics<br/>HuggingFace Sentiment"]
    end

    subgraph EXTRACTION["STAGE 3 — Structured Output CSVs"]
        direction LR
        E1["audio_output.csv<br/><code>Call_ID, Transcript,<br/>Extracted_Event, Location,<br/>Sentiment, Urgency_Score</code>"]
        E2["pdf_output.csv<br/><code>Report_ID, Incident_Type,<br/>Date, Location,<br/>Officer, Summary</code>"]
        E3["image_output.csv<br/><code>Image_ID, Scene_Type,<br/>Objects_Detected,<br/>Text_Extracted, Confidence</code>"]
        E4["video_output.csv<br/><code>Timestamp, Frame_ID,<br/>Event_Detected,<br/>Objects, Confidence</code>"]
        E5["text_output.csv<br/><code>Text_ID, Source, Raw_Text,<br/>Sentiment, Entities,<br/>Topic</code>"]
    end

    subgraph INTEGRATION["STAGE 4 — Data Integration"]
        direction TB
        M1["Define Incident_ID<br/>across all 5 CSVs"]
        M2["Merge via pandas<br/>join on Incident_ID"]
        M3["Handle missing<br/>values per modality"]
        M4["Severity Classification<br/>Low / Medium / High"]
        M1 --> M2 --> M3 --> M4
    end

    subgraph DASHBOARD["STAGE 5 — Dashboard & Query"]
        D1["Streamlit Dashboard<br/>Filter by severity, type, location<br/>Visualize incident distribution"]
    end

    A1 --> P1 --> E1
    A2 --> P2 --> E2
    A3 --> P3 --> E3
    A4 --> P4 --> E4
    A5 --> P5 --> E5

    E1 --> INTEGRATION
    E2 --> INTEGRATION
    E3 --> INTEGRATION
    E4 --> INTEGRATION
    E5 --> INTEGRATION

    INTEGRATION --> DASHBOARD
```

---

## Team Roles & Pipeline Ownership

| Student | Role | Data Type | Key AI Tools | Output CSV |
|---------|------|-----------|-------------|------------|
| **Lokeshwar** (Student 1) | Audio Analyst | 911 emergency calls | Whisper, spaCy, HuggingFace | `audio_output.csv` |
| **Neha** (Student 2) | Document Analyst | Police report PDFs | PyMuPDF, spaCy NER, BART | `pdf_output.csv` |
| **Nagraj** (Student 3) | Image Analyst | Fire/scene photos | YOLOv8, ViT, Tesseract | `image_output.csv` |
| **Rahul** (Student 4) | Video Analyst | CCTV surveillance | OpenCV, YOLOv8, motion det. | `video_output.csv` |
| **Swet** (Student 5) | Text Analyst | Crime reports/news | NLTK, spaCy, HuggingFace | `text_output.csv` |

---

## Detailed Pipeline Flow Diagram

```mermaid
flowchart LR
    subgraph S1["Student 1 — Lokeshwar"]
        direction TB
        A_IN["WAV Audio Files"]
        A_W["Whisper STT"]
        A_NER["spaCy NER<br/>Events, Locations"]
        A_SENT["HuggingFace<br/>Sentiment + Urgency"]
        A_OUT["audio_output.csv"]
        A_IN --> A_W --> A_NER --> A_SENT --> A_OUT
    end

    subgraph S2["Student 2 — Neha"]
        direction TB
        B_IN["PDF Documents"]
        B_EXT["PyMuPDF / pdfplumber<br/>+ Tesseract OCR"]
        B_NER["spaCy NER<br/>Dates, Names, Locations"]
        B_SUM["BART Summarizer"]
        B_OUT["pdf_output.csv"]
        B_IN --> B_EXT --> B_NER --> B_SUM --> B_OUT
    end

    subgraph S3["Student 3 — Nagraj"]
        direction TB
        C_IN["Scene Images"]
        C_YOLO["YOLOv8<br/>Object Detection"]
        C_VIT["ViT Scene<br/>Classification"]
        C_OCR["Tesseract OCR<br/>Text in Images"]
        C_OUT["image_output.csv"]
        C_IN --> C_YOLO --> C_VIT --> C_OCR --> C_OUT
    end

    subgraph S4["Student 4 — Rahul"]
        direction TB
        D_IN["CCTV Video Clips"]
        D_FRM["OpenCV<br/>Frame Extraction"]
        D_MOT["Motion Detection<br/>Key Frame Selection"]
        D_DET["YOLOv8<br/>Object Detection"]
        D_EVT["Rule-Based<br/>Event Classification"]
        D_OUT["video_output.csv"]
        D_IN --> D_FRM --> D_MOT --> D_DET --> D_EVT --> D_OUT
    end

    subgraph S5["Student 5 — Swet"]
        direction TB
        E_IN["Crime Text / News"]
        E_PRE["NLTK Preprocess<br/>Tokenize, Clean"]
        E_NER["spaCy NER<br/>Entities"]
        E_TOP["Zero-Shot<br/>Topic Classification"]
        E_SENT["HuggingFace<br/>Sentiment Analysis"]
        E_OUT["text_output.csv"]
        E_IN --> E_PRE --> E_NER --> E_TOP --> E_SENT --> E_OUT
    end
```

---

## Integration Phase (Stage 4) — Final Merged Dataset

```mermaid
flowchart TB
    CSV1["audio_output.csv"] --> ADD_ID["Add Incident_ID<br/>to all CSVs"]
    CSV2["pdf_output.csv"] --> ADD_ID
    CSV3["image_output.csv"] --> ADD_ID
    CSV4["video_output.csv"] --> ADD_ID
    CSV5["text_output.csv"] --> ADD_ID

    ADD_ID --> MERGE["pandas merge/join<br/>on Incident_ID"]
    MERGE --> MISSING["Handle missing values<br/>NaN for absent modalities"]
    MISSING --> SEVERITY["Severity Classification<br/>Low / Medium / High<br/>based on combined signals"]
    SEVERITY --> FINAL["final_integrated_dataset.csv"]
    FINAL --> DASH["Streamlit Dashboard"]
```

### Final Integrated Output Schema

| Column | Source | Description |
|--------|--------|-------------|
| `Incident_ID` | Generated | Common key `INC_001, INC_002...` |
| `Audio_Event` | Audio CSV | Extracted event from 911 call |
| `PDF_Doc_Type` | PDF CSV | Incident type from police report |
| `Image_Objects` | Image CSV | Detected objects with confidence |
| `Video_Event` | Video CSV | Event detected in CCTV frame |
| `Text_Crime_Type` | Text CSV | Crime type from news/social media |
| `Severity` | Computed | Low / Medium / High from combined signals |

---

## Repository Structure

```
multimodal-incident-analyzer/
├── audio/                    # Student 1 — Lokeshwar
│   ├── audio_pipeline.py
│   ├── data/                 # 911 audio WAV files
│   ├── audio_output.csv
│   └── requirements.txt
├── pdf/                      # Student 2 — Neha
│   ├── pdf_pipeline.py
│   ├── data/                 # MuckRock LESO2.pdf
│   ├── pdf_output.csv
│   └── requirements.txt
├── images/                   # Student 3 — Nagraj
│   ├── image_pipeline.py
│   ├── data/                 # D-Fire / Roboflow images
│   ├── image_output.csv
│   └── requirements.txt
├── video/                    # Student 4 — Rahul
│   ├── video_pipeline.py
│   ├── data/CAVIAR/          # CAVIAR CCTV clips
│   ├── video_output.csv
│   └── requirements.txt
├── text/                     # Student 5 — Swet
│   ├── text_pipeline.py
│   ├── data/                 # CrimeReport dataset
│   ├── text_output.csv
│   └── requirements.txt
├── integration/              # Team — Final merge
│   ├── integrate.py
│   ├── dashboard.py          # Streamlit dashboard
│   └── final_integrated_dataset.csv
├── README.md
└── requirements.txt          # Master dependencies
```

---

## Datasets Summary

| Pipeline | Dataset | Source | Size |
|----------|---------|-------|------|
| Audio | 911 Calls + Wav2Vec2 | Kaggle | 12 audio samples |
| PDF | Arkansas PD 1033 Training Proposals | MuckRock FOIA | 75 pages, 5 reports |
| Image | D-Fire / Roboflow Fire Detection | Kaggle / Roboflow | 21,527 images |
| Video | CAVIAR CCTV Dataset | Univ. of Edinburgh | 10 clips, ~92 MB |
| Text | CrimeReport | Kaggle | 1,000+ articles |

---

## AI Models & Tools Used

```mermaid
mindmap
    root["AI Models & Tools"]
        Speech-to-Text
            OpenAI Whisper
            Wav2Vec2
        NLP
            spaCy NER
            NLTK Preprocessing
            HuggingFace Transformers
                Sentiment Analysis
                Zero-Shot Classification
                BART Summarization
        Computer Vision
            YOLOv8 Object Detection
            ViT Scene Classification
            Tesseract OCR
            OpenCV Frame Processing
        Integration
            pandas merge/join
            Severity Classifier
            Streamlit Dashboard
```

---

## Marking Rubric Alignment

| Criteria | Weight | How We Address It |
|----------|--------|-------------------|
| **Problem Understanding** | 10% | Clear role assignment per student, well-defined data sources |
| **Data Collection** | 15% | Real datasets: Kaggle 911, MuckRock PDF, Roboflow fire, CAVIAR CCTV, CrimeReport |
| **AI Model Implementation** | 25% | Whisper, YOLOv8, ViT, spaCy, HuggingFace transformers across all modalities |
| **Pipeline Design** | 15% | 5-stage architecture: Ingestion → Processing → Extraction → Integration → Dashboard |
| **Data Integration** | 15% | All 5 CSVs merged on Incident_ID with severity classification |
| **Code Quality** | 10% | Organized GitHub repo with /audio, /pdf, /images, /video, /text, /integration |
| **Final Demonstration** | 10% | End-to-end demo: raw data → structured output → dashboard visualization |
