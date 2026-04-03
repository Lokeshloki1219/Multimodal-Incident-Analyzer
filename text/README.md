# 📝 Text Pipeline — Student 5 (Swet Vimalkumar Patel)

## Overview

Classifies crime news articles and social media reports using zero-shot topic classification, named entity recognition, sentiment analysis, and severity scoring.

## Pipeline Stages

1. **Data Loading** — Reads JSONL crime report tweets and/or CSV news articles from `data/`
2. **Preprocessing** — Tokenization, stopword removal, text cleaning using NLTK
3. **Topic Classification** — HuggingFace zero-shot classifier (facebook/bart-large-mnli) with 12 crime categories
4. **Entity Extraction** — spaCy NER extracts locations, persons, organizations
5. **Sentiment Analysis** — Keyword-based and model-based sentiment (Negative/Neutral/Positive)
6. **Severity Classification** — Rule-based severity (High/Medium/Low) from keywords + topic

## Crime Categories

Murder/Homicide, Robbery/Theft, Assault/Violence, Drug Crime, Arson/Fire, Traffic Accident, Kidnapping/Missing Person, Fraud/White-collar Crime, Public Disturbance, Sexual Assault, Gang Violence, Shooting, Law Enforcement/Policing

## Dataset

- **Source**: CrimeReport Twitter dataset (JSONL format)
- **Format**: JSON Lines with `text`, `place`, `created_at` fields
- **Size**: 113 articles after processing

## Output

`text_output.csv` — 113 rows with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Text_ID` | String | Unique identifier (TXT_001, TXT_002, ...) |
| `Source` | String | Data source (CrimeReport, NewsArticle, etc.) |
| `Raw_Text` | String | Original text (truncated to 300 chars) |
| `Crime_Type` | String | Zero-shot classified crime category |
| `Location_Entity` | String | NER-extracted location or tweet place |
| `Sentiment` | String | Negative / Neutral / Positive |
| `Entities` | String | Extracted named entities summary |
| `Topic` | String | Classified topic category |
| `Severity_Label` | String | High / Medium / Low severity |

## Usage

```bash
python text/text_pipeline.py
```

## Dependencies

```
transformers
torch
spacy
nltk
pandas
numpy
```
