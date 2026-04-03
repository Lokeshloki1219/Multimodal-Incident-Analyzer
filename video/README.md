# 🎬 Video Pipeline — Student 4 (Rahul Sarma Vogeti)

## Overview

Analyzes CCTV surveillance footage by extracting key frames, detecting objects, and classifying events using a hybrid AI approach.

## Pipeline Stages

1. **Frame Extraction** — OpenCV samples 1 frame per second from video clips
2. **Motion Detection** — Frame differencing to compute motion scores and flag key frames
3. **Key Frame Selection** — Top-N frames by motion score, re-sorted chronologically
4. **Object Detection** — YOLOv8n with upscaling (CAVIAR footage is 384×288) and false-positive remapping
5. **Event Classification** — Rule-based classification using filename hints, person count, motion levels, and detected objects
6. **Scene Classification** — CLIP ViT zero-shot classification on key frames

## Dataset

- **Source**: CAVIAR CCTV dataset (University of Edinburgh)
- **URL**: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
- **Format**: MPEG video clips in `data/` subdirectory
- **Scenarios**: Browse, Fight, Collapse, Chase, LeftBag, Walk, Meet, Rest

## Output

`video_output.csv` — 198 rows with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Clip_ID` | String | Video clip identifier (CAVIAR_01, CAVIAR_02, ...) |
| `Timestamp` | String | Frame timestamp (MM:SS) |
| `Frame_ID` | String | Unique frame identifier (FRM_001, FRM_002, ...) |
| `Event_Detected` | String | Classified event (Fighting, Running, Loitering, etc.) |
| `Persons_Count` | Integer | Number of persons detected in frame |
| `Objects` | String | All detected objects with counts |
| `Confidence` | Float | 0.0–1.0 highest detection confidence |

## Usage

```bash
python video/video_pipeline.py
```

## Dependencies

```
ultralytics (YOLOv8)
opencv-python
transformers
torch
Pillow
pandas
numpy
```
