# 🔗 Integration Pipeline + Dashboard

## Overview

Combines all five modality CSVs into a unified incident dataset, generates AI summaries using Flan-T5, and provides an interactive Streamlit dashboard for querying and visualization. Includes a real-time file monitor for live incident ingestion.

## Components

### `integrate.py` — Sequential Merge Pipeline

Merges audio, PDF, image, video, and text outputs into `final_incidents.csv`:

- **Strategy**: Sequential alignment — row `i` of each modality maps to incident `INC_{i}`
- **Coverage**: 198 incidents (limited by 2nd-largest modality to ensure ≥2 sources per row)
- **Missing Values**: Context-aware descriptions ("No 911 call recorded") instead of NaN
- **Severity**: Adaptive scoring from Audio urgency, Image confidence, Text severity → High/Medium/Low

### `summarizer.py` — LLM Summarizer

Generates 2–3 sentence natural language summaries using **google/flan-t5-base**:

- Runs locally, no API key needed (~990MB download on first run)
- Processes ~1–2 rows/second on CPU
- Adds `AI_Summary` column to `final_incidents.csv`

### `dashboard.py` — Streamlit Dashboard

Interactive web dashboard at `http://localhost:8501`:

- **Metrics**: Total incidents, severity breakdown, average modality count
- **Charts**: Severity distribution, modality coverage, top audio events, top crime types
- **Filters**: Severity level, modality count, incident ID range
- **Detail View**: Per-incident breakdown with AI-generated summary
- **Data Explorer**: Raw data tabs for each modality CSV
- **Auto-Refresh**: 10-second TTL cache + manual refresh button for real-time updates

### `realtime_monitor.py` — Watchdog File Monitor

Watches `new_data/` folder for incoming files and automatically processes them:

- **File Detection**: Watchdog observer monitors for new files
- **Auto-Routing**: Extension-based routing (`.wav`→Audio, `.pdf`→PDF, `.jpg`→Image, etc.)
- **Incremental Updates**: Appends to modality CSV + creates new incident (no full re-merge)
- **Smart Summarization**: Runs Flan-T5 on the new row only
- **Concurrency**: File-based lock prevents race conditions
- **Archival**: Processed files are moved to `new_data/_processed/`

## Output Files

| File | Description |
|------|-------------|
| `final_incidents.csv` | 198 unified incidents (9 columns + AI_Summary) |
| `final_detailed_dataset.csv` | Extended dataset with additional modality fields |
| `final_integrated_dataset.csv` | Compact integrated format |
| `final_summary_dataset.csv` | Summary statistics per incident |

## Usage

```bash
# Step 1: Run integration merge
python integration/integrate.py

# Step 2: Generate AI summaries (optional, ~5 min on CPU)
python integration/summarizer.py

# Step 3: Launch dashboard
streamlit run integration/dashboard.py

# Real-time mode (2 terminals):
# Terminal 1:
python integration/realtime_monitor.py
# Terminal 2:
streamlit run integration/dashboard.py
# Then drop files into new_data/ folder
```

## Dependencies

```
pandas
numpy
streamlit
transformers
torch
watchdog
```
