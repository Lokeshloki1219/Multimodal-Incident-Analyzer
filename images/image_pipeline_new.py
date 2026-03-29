"""
============================================================
 AI for Engineers — Multimodal Crime / Incident Analyzer
 Student 3: Image Analyst — Nagraj V Vallakati
 Module   : images/image_pipeline.py
============================================================

Pipeline:
  1. Load images from dataset (test / train / valid splits)
  2. Train YOLOv8 on fire dataset OR load existing trained weights
  3. Run fire-specific object detection on all images
  4. Classify scene type from detected objects
  5. Extract visible text via pytesseract OCR
  6. Save structured output → images/image_output.csv

Usage:
  python images/image_pipeline.py                        # auto-trains then runs
  python images/image_pipeline.py --skip-train           # skip training, use existing best.pt
  python images/image_pipeline.py --epochs 20            # train for 20 epochs (default: 15)

Requirements:
  pip install -r images/requirements.txt
  apt-get install -y tesseract-ocr        # Linux / Colab
  brew install tesseract                  # macOS
  choco install tesseract                 # Windows
"""

import os
import glob
import cv2
import yaml
import numpy as np
import pandas as pd
import pytesseract
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Root of the images dataset (containing test/ train/ valid/ folders)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Output CSV path
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "image_output.csv")

# Where trained weights will be saved after training
TRAINED_WEIGHTS = os.path.join(os.path.dirname(__file__), "runs", "train", "weights", "best.pt")

# Base model to fine-tune from (downloads ~6MB on first run)
BASE_MODEL = "yolov8n.pt"

# Training config
EPOCHS     = 15      # increase for better accuracy (recommended: 20-50)
IMG_SIZE   = 640     # standard YOLOv8 input size
BATCH_SIZE = 8      # reduce to 8 if you get out-of-memory errors

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.25   # lowered from 0.40 — fire models need lower threshold

# Image extensions
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


# ══════════════════════════════════════════════════════════════════
#  SCENE CLASSIFICATION RULES
#  (covers both fire-dataset classes and generic COCO classes)
# ══════════════════════════════════════════════════════════════════

SCENE_RULES = {
    "Fire Scene":         ["fire", "smoke", "flame", "fire-smoke", "fire_smoke"],
    "Accident Scene":     ["car", "truck", "motorcycle", "bicycle", "bus", "vehicle", "ambulance"],
    "Theft / Robbery":    ["person", "backpack", "handbag", "knife"],
    "Public Disturbance": ["person", "crowd", "bottle"],
}


# ══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def find_data_yaml(dataset_dir: str) -> str:
    """
    Find the data.yaml file in the dataset folder.
    Roboflow datasets always include one — it defines class names and split paths.
    """
    candidates = [
        os.path.join(dataset_dir, "data.yaml"),
        os.path.join(dataset_dir, "dataset.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matches = glob.glob(os.path.join(dataset_dir, "**", "*.yaml"), recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"\n  No data.yaml found in: {dataset_dir}"
        "\n   Roboflow datasets always include a data.yaml — check your folder structure."
    )


def fix_yaml_paths(yaml_path: str, dataset_dir: str) -> str:
    """
    Roboflow data.yaml files sometimes use relative or absolute paths
    that don't match where you've placed the dataset locally.
    This rewrites the train/val/test paths to correct absolute paths.
    Returns path to the fixed yaml file.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    for split_key in ["train", "val", "test"]:
        if split_key in data:
            original    = data[split_key]
            folder_name = "valid" if split_key == "val" else split_key
            abs_path    = os.path.join(dataset_dir, folder_name, "images")
            if os.path.exists(abs_path):
                data[split_key] = abs_path
                log.info(f"   YAML '{split_key}': {original}  ->  {abs_path}")

    fixed_path = os.path.join(os.path.dirname(yaml_path), "data_fixed.yaml")
    with open(fixed_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    return fixed_path


def train_yolo(dataset_dir: str, epochs: int) -> tuple:
    """
    Fine-tune YOLOv8n on the fire detection dataset.
    Returns (path_to_best_weights, list_of_class_names).
    """
    log.info("\n" + "=" * 60)
    log.info("  TRAINING FIRE-SPECIFIC YOLOv8 MODEL")
    log.info("=" * 60)

    yaml_path = find_data_yaml(dataset_dir)
    log.info(f"\n  Found data.yaml: {yaml_path}")

    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    class_names = yaml_data.get("names", [])
    log.info(f"   Dataset classes: {class_names}")

    fixed_yaml = fix_yaml_paths(yaml_path, dataset_dir)
    log.info(f"   Fixed YAML saved: {fixed_yaml}")

    log.info(f"\n  Training YOLOv8n for {epochs} epochs...")
    log.info(f"   Image size : {IMG_SIZE}")
    log.info(f"   Batch size : {BATCH_SIZE}")
    log.info("   This will take 5-20 minutes depending on your hardware.\n")

    model   = YOLO(BASE_MODEL)
    results = model.train(
        data     = fixed_yaml,
        epochs   = epochs,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        project  = os.path.join(os.path.dirname(__file__), "runs"),
        name     = "train",
        exist_ok = True,
        verbose  = False,
    )

    best_weights = os.path.join(
        os.path.dirname(__file__), "runs", "train", "weights", "best.pt"
    )
    if not os.path.exists(best_weights):
        raise FileNotFoundError(
            f"Training completed but best.pt not found at: {best_weights}"
        )

    log.info(f"\n  Training complete! Best weights -> {best_weights}")
    return best_weights, class_names


def get_image_paths(dataset_dir: str) -> dict:
    """Collect image paths from test/, train/, valid/ sub-folders."""
    splits = {}
    for split in ["test", "train", "valid"]:
        images_dir = os.path.join(dataset_dir, split, "images")
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(images_dir, ext)))
        splits[split] = sorted(paths)
        log.info(f"   {split}/images  ->  {len(paths)} images")
    return splits


def classify_scene(detected_labels: list) -> str:
    """Rule-based scene classifier — maps detected objects to a scene category."""
    labels_lower = [label.lower() for label in detected_labels]
    for scene, keywords in SCENE_RULES.items():
        if any(kw in labels_lower for kw in keywords):
            return scene
    return "Unknown Scene"


def extract_ocr_text(image_path: str) -> str:
    """Extract visible text from image using pytesseract OCR."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "None"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        raw     = pytesseract.image_to_string(thresh, config="--psm 11")
        cleaned = " ".join(raw.split())
        return cleaned if len(cleaned) > 2 else "None"
    except Exception:
        return "None"


def run_yolo_inference(model: YOLO, image_path: str) -> tuple:
    """Run YOLOv8 inference. Returns (labels, bounding_box_strings, confidences)."""
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result  = results[0]

    detected_labels = []
    bounding_boxes  = []
    confidences     = []

    for box in result.boxes:
        label = model.names[int(box.cls)]
        conf  = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detected_labels.append(label)
        bounding_boxes.append(f"{label}:[{x1},{y1},{x2},{y2}]")
        confidences.append(round(conf, 3))

    return detected_labels, bounding_boxes, confidences


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(dataset_dir: str, output_csv: str,
                 skip_train: bool, epochs: int) -> pd.DataFrame:

    log.info("=" * 60)
    log.info("  IMAGE ANALYST PIPELINE — Starting")
    log.info("=" * 60)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"\n  Dataset folder not found: {dataset_dir}"
            "\n   Update DATASET_DIR at the top of image_pipeline.py"
        )

    # ── Step 1: Train or load model ───────────────────────────────
    if os.path.exists(TRAINED_WEIGHTS):
        if skip_train:
            log.info(f"\n  Skipping training — loading: {TRAINED_WEIGHTS}")
        else:
            log.info(f"\n  Found existing weights: {TRAINED_WEIGHTS}")
            log.info("   To retrain: delete the 'images/runs/' folder and re-run.")
        model = YOLO(TRAINED_WEIGHTS)

    else:
        log.info("\n  No trained weights found — starting training...")
        weights_path, _ = train_yolo(dataset_dir, epochs)
        model = YOLO(weights_path)

    log.info(f"\n   Loaded model classes: {list(model.names.values())}")

    # ── Step 2: Collect all images ────────────────────────────────
    log.info(f"\n  Collecting images from: {dataset_dir}")
    splits     = get_image_paths(dataset_dir)
    all_images = []
    split_map  = {}

    for split, paths in splits.items():
        all_images.extend(paths)
        for p in paths:
            split_map[p] = split

    total = len(all_images)
    log.info(f"\n  Total images to process: {total}")

    if total == 0:
        raise RuntimeError(
            "\n  No images found! Expected structure:\n"
            "     dataset/test/images/*.jpg\n"
            "     dataset/train/images/*.jpg\n"
            "     dataset/valid/images/*.jpg"
        )

    # ── Step 3: Run inference on every image ──────────────────────
    log.info(f"\n  Running inference on {total} images...\n")
    results_list = []

    for idx, img_path in enumerate(all_images):

        image_id = f"IMG_{idx + 1:04d}"
        filename = os.path.basename(img_path)
        split    = split_map[img_path]

        detected_labels, bounding_boxes, confidences = run_yolo_inference(model, img_path)
        scene_type = classify_scene(detected_labels)
        ocr_text   = extract_ocr_text(img_path)
        avg_conf   = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        results_list.append({
            "Image_ID":         image_id,
            "Filename":         filename,
            "Split":            split,
            "Scene_Type":       scene_type,
            "Objects_Detected": ", ".join(sorted(set(detected_labels))) if detected_labels else "None",
            "Bounding_Boxes":   " | ".join(bounding_boxes) if bounding_boxes else "None",
            "Text_Extracted":   ocr_text,
            "Confidence_Score": avg_conf,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            log.info(
                f"  [{idx+1:>4}/{total}]  {image_id}  |  {split:<5}  |  "
                f"Scene: {scene_type:<22}  |  "
                f"Objects: {list(set(detected_labels))}  |  Conf: {avg_conf}"
            )

    # ── Step 4: Save results ──────────────────────────────────────
    df = pd.DataFrame(results_list)

    log.info("\n" + "=" * 60)
    log.info("  RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total images processed  : {len(df)}")
    log.info(f"  Average confidence      : {df['Confidence_Score'].mean():.3f}")
    log.info(f"  Images with OCR text    : {len(df[df['Text_Extracted'] != 'None'])}")
    log.info("\n  Scene Type Distribution:")
    for scene, count in df["Scene_Type"].value_counts().items():
        bar = "█" * min(count, 25)
        log.info(f"    {scene:<25}  {bar}  ({count})")

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    log.info(f"\n  Output saved -> {output_csv}")
    log.info(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    log.info(f"   Columns: {list(df.columns)}")
    log.info("\n  Image pipeline complete!\n")

    return df


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Image Analyst Pipeline — Multimodal Incident Analyzer"
    )
    parser.add_argument("--dataset",    type=str,   default=DATASET_DIR,
                        help="Path to dataset folder (must contain test/train/valid)")
    parser.add_argument("--output",     type=str,   default=OUTPUT_CSV,
                        help="Path for output CSV file")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--conf",       type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load existing runs/train/weights/best.pt")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf

    df = run_pipeline(
        dataset_dir = args.dataset,
        output_csv  = args.output,
        skip_train  = args.skip_train,
        epochs      = args.epochs,
    )

    print("\nFirst 5 rows of output:")
    print(df.head(5).to_string(index=False))
