# 🎙️ Audio Pipeline — Student 1 (Lokeshwar Reddy Peddarangareddy)

## Overview

Processes 911 emergency audio calls into structured incident data using a multi-stage AI pipeline.

## Pipeline Stages

1. **Speech-to-Text** — OpenAI Whisper (base model) transcribes `.wav` files to text
2. **Entity Extraction** — spaCy NER extracts locations, events, and persons from transcripts
3. **Sentiment Analysis** — HuggingFace distilbert-based sentiment classifier
4. **Urgency Scoring** — Weighted keyword matching (0–1 scale) based on emergency terms

## Dataset

- **Source**: 911 emergency call audio recordings
- **Format**: WAV files in `data/` subdirectory
- **Size**: 703 audio files
- **Download**: Run `python download_dataset.py` to fetch the dataset

## Output

`audio_output.csv` — 703 rows with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Call_ID` | String | Unique identifier (CALL_001, CALL_002, ...) |
| `Transcript` | String | Full text transcription from Whisper |
| `Extracted_Event` | String | Classified event type (Fire, Shooting, Robbery, etc.) |
| `Location` | String | Extracted location from NER (or "Unknown") |
| `Sentiment` | String | Negative / Neutral / Positive |
| `Urgency_Score` | Float | 0.0–1.0 urgency level from keyword weights |

## Usage

```bash
# Process all audio files
python audio/audio_pipeline.py

# Download the dataset first (if data/ is empty)
python audio/download_dataset.py
```

## Dependencies

```
openai-whisper
torch
transformers
spacy
pandas
numpy
```

## AI Models

- **OpenAI Whisper** (base, ~140MB) — Speech recognition
- **spaCy en_core_web_sm** — Named Entity Recognition
- **HuggingFace Transformers** — Sentiment classification
