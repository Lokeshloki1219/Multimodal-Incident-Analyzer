# 🖼️ Image Pipeline — Student 3 (Nagraj V Vallakati)

## Overview

Analyzes crime scene and incident images using a **two-model hybrid** object detection approach with AI-powered scene classification and OCR text extraction.

## Pipeline Stages

1. **Model 1 — Fire/Smoke Detector**: Fine-tuned YOLOv8n trained on Roboflow "Fire Smoke and Human Detector" dataset (3 classes: fire, smoke, human)
2. **Model 2 — COCO Detector**: Standard YOLOv8n pre-trained on COCO (80 classes: person, car, truck, knife, etc.)
3. **Detection Merging**: Fire/smoke from Model 1; person count = max(Model 1 human, Model 2 person); all other objects from Model 2
4. **Scene Classification**: HuggingFace ViT (google/vit-base-patch16-224) with keyword fallback
5. **OCR Text Extraction**: pytesseract extracts visible text from images (signs, license plates, etc.)
6. **Confidence Scoring**: Average detection confidence across both models

## Dataset

- **Source**: Roboflow "Fire Smoke and Human Detector" dataset (YOLOv8 format)
- **Format**: JPG images with YOLO annotation labels in `data/` subdirectory
- **Selection**: 80% labeled images (fire/smoke) + 20% backgrounds = 100 images

## Output

`image_output.csv` — 100 rows with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Image_ID` | String | Unique identifier (IMG_001, IMG_002, ...) |
| `Scene_Type` | String | Classified scene (Fire Scene, Vehicle Accident, etc.) |
| `Objects_Detected` | String | Comma-separated detected objects with counts |
| `Bounding_Boxes` | String | Detection region descriptions |
| `Text_Extracted` | String | OCR-extracted visible text |
| `Confidence_Score` | Float | 0.0–1.0 average detection confidence |

## Usage

```bash
# Train the fire/smoke detection model (first time only)
python images/train_fire_model.py

# Run the full image pipeline
python images/image_pipeline.py
```

## Dependencies

```
ultralytics (YOLOv8)
transformers
torch
pytesseract
Pillow
pandas
numpy
```
