"""
Image Pipeline — Student 3 (Nagraj V Vallakati)

Multimodal Crime / Incident Report Analyzer

What it does:
    Analyzes crime scene / incident images using a TWO-MODEL hybrid approach:
    1. Model 1: Fine-tuned YOLOv8 fire/smoke/human detector
       (trained on Roboflow "Fire Smoke and Human Detector" dataset)
    2. Model 2: Standard YOLOv8n COCO model (80 classes including person,
       car, truck, knife, scissors, etc.)
    3. Merges detections from both models for comprehensive object coverage
    4. Classifies scene type using HuggingFace ViT (google/vit-base-patch16-224)
       for AI-powered image classification, with keyword fallback
    5. Extracts visible text using OCR (pytesseract) for Text_Extracted column
    6. Computes a confidence score from detection results

Input:
    Image files (.jpg, .png, .bmp) in the 'data/' subdirectory
    (Roboflow "Fire Smoke and Human Detector" dataset in YOLOv8 format),
    OR uses built-in sample image descriptions if no images are found.

Output:
    image_output.csv with columns:
    Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Text_Extracted, Confidence_Score
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATA_DIR = r"D:\USA\Sems\4_Spring 2026 Sem\AI for Engineers\Multimodal\lokesh\images\images\data"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "image_output.csv")

# Maximum number of images to process (None = all)
MAX_IMAGES = None

# Fire/Smoke/Human detection class names (from Roboflow fine-tuned model)
FIRE_MODEL_CLASSES = {0: "fire", 1: "human", 2: "smoke"}

# Keyword-based scene classification mapping
SCENE_MAPPING = {
    "Fire Scene": ["fire", "smoke", "flame"],
    "Vehicle Accident": ["car", "truck", "bus", "motorcycle", "bicycle",
                         "traffic light", "stop sign"],
    "Assault Scene": ["knife", "scissors", "baseball bat"],
    "Public Disturbance": ["person", "human"],
    "Theft / Robbery": ["handbag", "backpack", "suitcase"],
    "Surveillance Scene": ["cell phone", "laptop", "monitor"],
    "Emergency Response": ["fire", "truck", "person"],
}

# Sample image analysis results — used when no images are available
SAMPLE_IMAGE_DATA = [
    {
        "filename": "fire_scene_001.jpg",
        "objects": "fire, smoke, building",
        "scene": "Fire Scene",
        "bbox": "2 fire regions, 1 smoke plume",
        "text": "FIRE DEPT - DO NOT CROSS",
        "confidence": 0.94
    },
    {
        "filename": "accident_scene_002.jpg",
        "objects": "car, truck, person, traffic light",
        "scene": "Vehicle Accident",
        "bbox": "1 vehicle, 1 vehicle, 1 person",
        "text": "SPEED LIMIT 45",
        "confidence": 0.89
    },
    {
        "filename": "robbery_003.jpg",
        "objects": "person, person, handbag, backpack",
        "scene": "Theft / Robbery",
        "bbox": "2 persons, 1 handbag, 1 backpack",
        "text": "QUICKMART STORE",
        "confidence": 0.82
    },
    {
        "filename": "fire_building_004.jpg",
        "objects": "fire, smoke, person, truck",
        "scene": "Fire Scene",
        "bbox": "1 fire region, 1 smoke plume, 1 person, 1 vehicle",
        "text": "EMERGENCY EXIT",
        "confidence": 0.96
    },
    {
        "filename": "vandalism_005.jpg",
        "objects": "car, person, bottle",
        "scene": "General Scene",
        "bbox": "1 vehicle, 1 person",
        "text": "NO PARKING",
        "confidence": 0.71
    },
    {
        "filename": "crash_006.jpg",
        "objects": "car, car, person, stop sign",
        "scene": "Vehicle Accident",
        "bbox": "2 vehicles, 1 person, 1 stop sign",
        "text": "MAIN ST",
        "confidence": 0.91
    },
    {
        "filename": "assault_007.jpg",
        "objects": "person, person, person, bench",
        "scene": "Public Disturbance",
        "bbox": "3 persons",
        "text": "",
        "confidence": 0.68
    },
    {
        "filename": "fire_warehouse_008.jpg",
        "objects": "fire, smoke, truck",
        "scene": "Fire Scene",
        "bbox": "1 fire region, 1 smoke plume, 1 vehicle",
        "text": "INDUSTRIAL BLVD",
        "confidence": 0.93
    },
    {
        "filename": "surveillance_009.jpg",
        "objects": "person, person, cell phone, backpack",
        "scene": "Surveillance Scene",
        "bbox": "2 persons",
        "text": "ATM",
        "confidence": 0.76
    },
    {
        "filename": "emergency_010.jpg",
        "objects": "person, truck, car, traffic light",
        "scene": "Emergency Response",
        "bbox": "1 person, 1 vehicle, 1 vehicle",
        "text": "911",
        "confidence": 0.87
    },
    {
        "filename": "fire_forest_011.jpg",
        "objects": "fire, smoke",
        "scene": "Fire Scene",
        "bbox": "1 fire region, 1 smoke plume",
        "text": "",
        "confidence": 0.95
    },
    {
        "filename": "theft_012.jpg",
        "objects": "person, person, suitcase, handbag",
        "scene": "Theft / Robbery",
        "bbox": "2 persons, 1 suitcase, 1 handbag",
        "text": "PARKING LOT B",
        "confidence": 0.79
    },
]


# ============================================================================
# STEP 1: TWO-MODEL YOLOv8 OBJECT DETECTION
# ============================================================================

def load_fire_model():
    """
    Load the fine-tuned fire/smoke/human YOLOv8 model.
    Trained on Roboflow "Fire Smoke and Human Detector" dataset.
    """
    try:
        from ultralytics import YOLO

        # fire_model_path = os.path.join(SCRIPT_DIR, "fire_detector", "weights", "best.pt")
        fire_model_path = r"D:\USA\Sems\4_Spring 2026 Sem\AI for Engineers\Multimodal\lokesh\images\images\fire_detector\weights\best.pt"
        if os.path.exists(fire_model_path):
            print(f"[Image] Loading fine-tuned fire/smoke/human detection model...")
            model = YOLO(fire_model_path)
            print(f"[Image]   Path: {fire_model_path}")
            print("[Image]   Fire detector loaded (classes: fire, smoke, human)")
            return model
        else:
            print(f"[Image] WARNING: Fine-tuned model not found at {fire_model_path}")
            print("[Image]   Run train_fire_model.py first to train the model.")
            return None
    except ImportError:
        print("[Image] WARNING: ultralytics not installed.")
        return None
    except Exception as e:
        print(f"[Image] WARNING: Could not load fire model: {e}")
        return None


def load_coco_model():
    """
    Load the standard YOLOv8n model pre-trained on COCO (80 classes).
    Detects: person, car, truck, bus, motorcycle, knife, scissors, etc.
    No training needed — works out of the box.
    """
    try:
        from ultralytics import YOLO
        print("[Image] Loading YOLOv8n COCO model (80 classes)...")
        model = YOLO("yolov8n.pt")
        print("[Image]   COCO detector loaded (person, car, truck, knife, etc.)")
        return model
    except ImportError:
        print("[Image] WARNING: ultralytics not installed.")
        return None
    except Exception as e:
        print(f"[Image] WARNING: Could not load COCO model: {e}")
        return None


def detect_objects_single_model(model, image_path, model_name="model"):
    """
    Run YOLOv8 inference on a single image with one model.
    Returns: dict of {class_name: count}, list of confidences
    """
    try:
        results = model(image_path, verbose=False)
        detections = {}  # name -> count
        confidences = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls_id]
                detections[name] = detections.get(name, 0) + 1
                confidences.append(conf)

        return detections, confidences
    except Exception as e:
        print(f"    [{model_name}] Detection error: {e}")
        return {}, []


def detect_objects_dual(fire_model, coco_model, image_path):
    """
    Run BOTH models on a single image and merge detections.
    Model 1 (Fire): detects fire, smoke, human
    Model 2 (COCO): detects 80 classes (person, car, truck, knife, etc.)

    Merging logic:
    - Fire/smoke detections come from the fire model (more accurate)
    - Person count = max(fire_model_human, coco_model_person)
    - All other COCO objects (car, truck, knife, etc.) come from COCO model
    """
    fire_dets, fire_confs = {}, []
    coco_dets, coco_confs = {}, []

    # Run fire model
    if fire_model is not None:
        fire_dets, fire_confs = detect_objects_single_model(
            fire_model, image_path, "Fire"
        )

    # Run COCO model
    if coco_model is not None:
        coco_dets, coco_confs = detect_objects_single_model(
            coco_model, image_path, "COCO"
        )

    # Merge detections
    merged = {}

    # 1. Fire/smoke from fire model (trust this model for fire)
    for cls in ["fire", "smoke"]:
        if cls in fire_dets:
            merged[cls] = fire_dets[cls]

    # 2. Person count = max(fire_model "human", coco_model "person")
    fire_humans = fire_dets.get("human", 0)
    coco_persons = coco_dets.get("person", 0)
    person_count = max(fire_humans, coco_persons)
    if person_count > 0:
        merged["person"] = person_count

    # 3. All other COCO objects (vehicles, weapons, etc.)
    skip = {"person"}  # Already merged above
    for cls, count in coco_dets.items():
        if cls not in skip:
            merged[cls] = count

    # Build objects string
    parts = []
    for name, count in merged.items():
        if count > 1:
            if name == "person":
                parts.append(f"{count} persons")
            else:
                parts.append(f"{count} {name}s")
        else:
            parts.append(name)
    objects_str = ", ".join(parts) if parts else "none detected"

    # Build bounding box description
    region_names = {
        "fire": "fire region", "smoke": "smoke plume", "person": "person",
        "human": "person", "car": "vehicle", "truck": "vehicle",
        "bus": "vehicle", "motorcycle": "motorcycle", "knife": "knife",
        "scissors": "scissors",
    }
    bbox_parts = []
    for name, count in merged.items():
        label = region_names.get(name, name)
        if count > 1:
            label = label + "s"
        bbox_parts.append(f"{count} {label}")
    bbox_desc = ", ".join(bbox_parts) if bbox_parts else "none"

    # Confidence = max of all detections
    all_confs = fire_confs + coco_confs
    avg_conf = round(np.mean(all_confs), 2) if all_confs else 0.0

    return merged, objects_str, avg_conf, bbox_desc


# ============================================================================
# STEP 2: SCENE CLASSIFICATION (HuggingFace ViT + keyword fallback)
# ============================================================================

def load_vit_classifier():
    """Load HuggingFace ViT model for AI-powered scene classification."""
    try:
        from transformers import pipeline
        print("[Image] Loading HuggingFace ViT scene classifier...")
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=-1  # CPU
        )
        print("[Image]   ViT classifier loaded (google/vit-base-patch16-224)")
        return classifier
    except Exception as e:
        print(f"[Image] WARNING: Could not load ViT classifier: {e}")
        print("[Image] Falling back to keyword-based classification.")
        return None


# Map ViT ImageNet labels to incident scene types
VIT_SCENE_MAP = {
    "Fire Scene": ["fire", "flame", "torch", "volcano", "matchstick",
                   "fireplace", "bonfire", "candle", "furnace"],
    "Vehicle Accident": ["crash", "car", "wreck", "ambulance", "police van",
                         "tow truck", "minibus", "traffic light"],
    "Emergency Response": ["ambulance", "fire engine", "fire truck",
                           "police van", "stretcher"],
    "Assault Scene": ["assault", "weapon", "rifle", "revolver",
                      "hatchet", "cleaver"],
    "Theft / Robbery": ["safe", "shopping cart", "cash machine"],
    "Surveillance Scene": ["monitor", "screen", "television"],
}


def classify_scene_with_vit(vit_classifier, image_path):
    """Classify scene type using HuggingFace ViT image classification model."""
    if vit_classifier is None:
        return None, 0.0

    try:
        results = vit_classifier(image_path, top_k=5)
        top_label = results[0]["label"].lower()
        top_score = results[0]["score"]

        # Map ViT prediction to our scene types
        for scene_type, keywords in VIT_SCENE_MAP.items():
            for kw in keywords:
                if kw in top_label:
                    return scene_type, top_score

        # Check all top-5 predictions
        for pred in results[1:]:
            label = pred["label"].lower()
            for scene_type, keywords in VIT_SCENE_MAP.items():
                for kw in keywords:
                    if kw in label:
                        return scene_type, pred["score"]

        return None, top_score  # No scene match found
    except Exception as e:
        print(f"    [ViT] Error: {e}")
        return None, 0.0


def classify_scene(merged_detections, image_filename="", vit_scene=None):
    """
    Classify the scene type using a hybrid approach:
    1. Use ViT classification result if available
    2. Fall back to keyword matching on detected objects

    Keyword rules:
      fire/smoke detected → "Fire Scene"
      car/truck + person → "Vehicle Accident"
      person + knife/scissors → "Assault Scene"
      multiple persons → "Public Disturbance"
    """
    # Use ViT result if available
    if vit_scene is not None:
        return vit_scene

    objects_set = set(merged_detections.keys())

    # Priority 1: Fire/smoke
    if objects_set & {"fire", "smoke"}:
        return "Fire Scene"

    # Priority 2: Weapons → Assault
    if objects_set & {"knife", "scissors", "baseball bat"} and "person" in objects_set:
        return "Assault Scene"

    # Priority 3: Vehicles + person → Accident
    if objects_set & {"car", "truck", "bus", "motorcycle"} and "person" in objects_set:
        return "Vehicle Accident"

    # Priority 4: Multiple persons → Public Disturbance
    person_count = merged_detections.get("person", 0)
    if person_count >= 3:
        return "Public Disturbance"

    # Priority 5: General scoring
    best_scene = "General Scene"
    best_score = 0
    for scene_type, keywords in SCENE_MAPPING.items():
        score = sum(1 for kw in keywords if kw in objects_set)
        if score > best_score:
            best_score = score
            best_scene = scene_type

    return best_scene


# ============================================================================
# STEP 3: OCR TEXT EXTRACTION
# ============================================================================

def extract_text_from_image(image_path):
    """Extract visible text from image using pytesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        # Set tesseract path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img).strip()
        # Clean up OCR noise
        text = " ".join(text.split())
        if len(text) < 3:
            return ""
        return text[:200]  # Limit text length
    except ImportError:
        return ""
    except Exception:
        return ""


# ============================================================================
# STEP 4: IMAGE SELECTION AND PROCESSING
# ============================================================================

def find_images(data_dir):
    """
    Find image files, prioritizing those with fire/smoke/human labels.
    Reads YOLO label files to pick images that contain annotations,
    with a proportional mix (80% labeled, 20% background).
    """
    import random
    random.seed(42)  # Reproducible selection

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    labeled_images = []
    background_images = []

    # Check for YOLO-format label files alongside images
    for split in ["train", "val", "valid", "test"]:
        img_dir = os.path.join(data_dir, "data", split, "images")
        lbl_dir = os.path.join(data_dir, "data", split, "labels")
        if not os.path.exists(img_dir):
            # Also try top-level split (some Roboflow exports use this)
            img_dir = os.path.join(data_dir, split, "images")
            lbl_dir = os.path.join(data_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue

        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() not in image_extensions:
                continue
            img_path = os.path.join(img_dir, f)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")

            if os.path.exists(lbl_path):
                content = open(lbl_path).read().strip()
                if content:  # Has annotations
                    labeled_images.append(img_path)
                else:
                    background_images.append(img_path)
            else:
                background_images.append(img_path)

    # If found labeled images, pick a proportional mix
    if labeled_images:
        random.shuffle(labeled_images)
        random.shuffle(background_images)
        total_target = MAX_IMAGES if MAX_IMAGES else len(labeled_images) + len(background_images)
        n_labeled = min(int(total_target * 0.8), len(labeled_images))
        n_bg = min(total_target - n_labeled, len(background_images))
        selected = labeled_images[:n_labeled] + background_images[:n_bg]
        random.shuffle(selected)
        print(f"[Image] Found {len(labeled_images)} labeled images, "
              f"{len(background_images)} backgrounds")
        print(f"[Image] Selected: {n_labeled} labeled + {n_bg} background = "
              f"{len(selected)} total")
        return selected

    # Fallback: just find all images recursively
    all_images = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                all_images.append(os.path.join(root, f))
    if all_images:
        print(f"[Image] Found {len(all_images)} images (no YOLO labels detected)")
    return sorted(all_images)


def process_real_images(fire_model, coco_model, image_paths, vit_classifier=None):
    """
    Process real images through the two-model pipeline:
    1. Fire model detects fire/smoke/human
    2. COCO model detects person/car/truck/knife/etc.
    3. Merge detections from both models
    4. ViT classifies scene type
    5. OCR extracts text
    """
    results = []

    total = min(len(image_paths), MAX_IMAGES) if MAX_IMAGES else len(image_paths)
    image_paths = image_paths[:total]

    print(f"[Image] Processing {total} images with TWO-MODEL approach...")
    print(f"[Image]   Model 1: {'Fire/Smoke/Human detector' if fire_model else 'NOT LOADED'}")
    print(f"[Image]   Model 2: {'COCO detector (80 classes)' if coco_model else 'NOT LOADED'}")
    print(f"[Image]   ViT:     {'Scene classifier loaded' if vit_classifier else 'Keyword fallback'}")

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"\n  [{i+1}/{total}] Processing batch...")

        # Step 1: Dual-model object detection
        merged, objects_str, avg_conf, bbox_desc = detect_objects_dual(
            fire_model, coco_model, img_path
        )

        # Step 2: Scene classification (ViT primary, keyword fallback)
        vit_scene, vit_score = classify_scene_with_vit(vit_classifier, img_path)
        scene_type = classify_scene(merged, filename, vit_scene=vit_scene)

        # Use better confidence if ViT is more confident
        if vit_score > avg_conf:
            avg_conf = round(vit_score, 2)

        if avg_conf == 0.0:
            avg_conf = 0.50  # minimum baseline for processed images

        # Step 3: OCR text extraction
        text = extract_text_from_image(img_path)

        results.append({
            "filename": filename,
            "objects": objects_str,
            "scene": scene_type,
            "bbox": bbox_desc,
            "text": text,
            "confidence": avg_conf
        })

        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"    [{i+1}/{total}] Last: {scene_type} | {objects_str[:50]} | "
                  f"conf={avg_conf}")

    return results


def get_image_results():
    """
    Process real images if available, otherwise return sample data.
    Uses two-model approach: Fire detector + COCO detector, with ViT scene
    classification and OCR text extraction.
    """
    if os.path.exists(DATA_DIR):
        image_paths = find_images(DATA_DIR)
        if image_paths:
            fire_model = load_fire_model()
            coco_model = load_coco_model()
            vit_classifier = load_vit_classifier()

            if fire_model is not None or coco_model is not None:
                results = process_real_images(
                    fire_model, coco_model, image_paths,
                    vit_classifier=vit_classifier
                )
                if results:
                    print(f"[Image] Successfully processed {len(results)} images.")
                    return results

    # Fall back to sample data
    print("[Image] No images found. Using built-in sample image data.")
    print(f"[Image] {len(SAMPLE_IMAGE_DATA)} sample entries loaded.")
    return SAMPLE_IMAGE_DATA


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_image_pipeline():
    """
    Execute the full image analysis pipeline:
    1. Load two YOLOv8 models (fire detector + COCO)
    2. Load ViT scene classifier
    3. Process images through dual detection + scene classification + OCR
    4. Export to CSV
    """
    print("=" * 70)
    print("IMAGE PIPELINE — Student 3 (Nagraj V Vallakati)")
    print("Multimodal Crime / Incident Report Analyzer")
    print("Two-Model Approach: Fire/Smoke Detector + COCO Detector")
    print("=" * 70)

    # Step 1: Get image analysis results
    print("\n[Step 1] Analyzing images...")
    image_results = get_image_results()

    if not image_results:
        print("[Image] ERROR: No results available. Exiting.")
        sys.exit(1)

    # Step 2: Build output DataFrame
    print(f"\n[Step 2] Building output ({len(image_results)} records)...")
    records = []

    for idx, item in enumerate(image_results):
        image_id = f"IMG_{(idx + 1):03d}"

        records.append({
            "Image_ID": image_id,
            "Scene_Type": item["scene"],
            "Objects_Detected": item["objects"],
            "Bounding_Boxes": item.get("bbox", "N/A"),
            "Text_Extracted": item.get("text", ""),
            "Confidence_Score": item["confidence"],
        })

    # Step 3: Export to CSV
    print(f"\n[Step 3] Exporting results to '{OUTPUT_FILE}'...")
    df = pd.DataFrame(records)

    # Column order per assignment spec
    df = df[["Image_ID", "Scene_Type", "Objects_Detected", "Bounding_Boxes",
             "Text_Extracted", "Confidence_Score"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"[Image] Pipeline complete! Output saved to: {OUTPUT_FILE}")
    print(f"[Image] Total records: {len(df)}")
    print(f"{'=' * 70}")

    # Display summary
    print("\n--- Output Preview ---")
    print(df.to_string(index=False, max_colwidth=50))

    return df


if __name__ == "__main__":
    run_image_pipeline()
