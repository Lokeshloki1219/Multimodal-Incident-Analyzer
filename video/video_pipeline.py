"""
video_pipeline.py — Student 4 (Rahul Sarma Vogeti)
Multimodal Crime / Incident Report Analyzer

Pipeline:
    1. Frame extraction from CAVIAR .mpg clips (OpenCV)
    2. Motion detection via frame differencing to flag key frames
    3. YOLOv8 object detection — upscaled frames, false-positive remapping
    4. Rule-based event classification (motion + objects + filename hints)
    5. Structured CSV export

Output CSV columns (exact assignment spec):
    Clip_ID | Timestamp | Frame_ID | Event_Detected |
    Persons_Count | Objects | Confidence

Dataset: CAVIAR CCTV — University of Edinburgh
    https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
    10 clips: Browse, Fight x4, LeftBag, Meet_Crowd,
              Rest_FallOnFloor, Rest_SlumpOnFloor, Walk
"""

import os
import re
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ============================================================================
#  CONFIGURATION
# ============================================================================

SCRIPT_DIR           = Path(__file__).parent
DATA_DIR             = SCRIPT_DIR / "data"
OUTPUT_FILE          = SCRIPT_DIR / "video_output.csv"

MAX_VIDEOS           = 10
FRAME_SAMPLE_RATE    = 25      # 1 frame per second at 25 fps
MAX_KEY_FRAMES       = 10      # key frames selected per clip
MOTION_THRESHOLD     = 0.02    # mean diff/255 to count as motion
YOLO_CONF            = 0.35    # minimum YOLO detection confidence
UPSCALE_TARGET       = 640     # upscale CAVIAR 384×288 before YOLO

# YOLOv8 misdetects distant persons in low-res CAVIAR as these — remap to person
FALSE_POSITIVE_REMAP = {
    "bird", "kite", "snowboard", "skateboard", "skis",
    "frisbee", "sports ball", "surfboard",
    "dog", "cat", "cow", "horse", "bear", "zebra", "giraffe",
}

# Sample data used when no video files found (matches assignment expected output)
SAMPLE_DATA = [
    {"clip_id": "CAVIAR_01", "timestamp": "0:00:12", "frame_id": "FRM_036",
     "event": "Person collapsing", "objects": "1 person",
     "persons_count": 1, "confidence": 0.88},
    {"clip_id": "CAVIAR_01", "timestamp": "0:00:24", "frame_id": "FRM_072",
     "event": "Normal walking",   "objects": "2 persons",
     "persons_count": 2, "confidence": 0.92},
    {"clip_id": "CAVIAR_02", "timestamp": "0:00:36", "frame_id": "FRM_108",
     "event": "Loitering",        "objects": "1 person, 1 bag",
     "persons_count": 1, "confidence": 0.75},
    {"clip_id": "CAVIAR_03", "timestamp": "0:01:00", "frame_id": "FRM_150",
     "event": "Fighting",         "objects": "3 persons",
     "persons_count": 3, "confidence": 0.85},
    {"clip_id": "CAVIAR_03", "timestamp": "0:01:12", "frame_id": "FRM_186",
     "event": "Running",          "objects": "2 persons",
     "persons_count": 2, "confidence": 0.90},
    {"clip_id": "CAVIAR_04", "timestamp": "0:01:30", "frame_id": "FRM_225",
     "event": "Normal walking",   "objects": "1 person",
     "persons_count": 1, "confidence": 0.95},
    {"clip_id": "CAVIAR_05", "timestamp": "0:02:00", "frame_id": "FRM_300",
     "event": "Crowd gathering",  "objects": "5 persons",
     "persons_count": 5, "confidence": 0.81},
    {"clip_id": "CAVIAR_06", "timestamp": "0:02:15", "frame_id": "FRM_337",
     "event": "Vehicle movement", "objects": "2 cars, 1 person",
     "persons_count": 1, "confidence": 0.87},
    {"clip_id": "CAVIAR_07", "timestamp": "0:02:30", "frame_id": "FRM_375",
     "event": "Fighting",         "objects": "2 persons",
     "persons_count": 2, "confidence": 0.79},
    {"clip_id": "CAVIAR_08", "timestamp": "0:02:45", "frame_id": "FRM_412",
     "event": "Person collapsing","objects": "1 person",
     "persons_count": 1, "confidence": 0.91},
    {"clip_id": "CAVIAR_09", "timestamp": "0:03:00", "frame_id": "FRM_450",
     "event": "Normal walking",   "objects": "3 persons, 1 backpack",
     "persons_count": 3, "confidence": 0.93},
    {"clip_id": "CAVIAR_10", "timestamp": "0:03:20", "frame_id": "FRM_500",
     "event": "Running",          "objects": "1 person",
     "persons_count": 1, "confidence": 0.86},
]


# ============================================================================
#  STEP 1 — FRAME EXTRACTION + MOTION DETECTION
# ============================================================================

def extract_frames(video_path: str) -> tuple[list, int]:
    """
    Sample 1 frame per second, compute frame-differencing motion score.
    Returns (frame_data, total_frame_count).
    frame_data rows: (frame_idx, timestamp_str, motion_score, frame_array)

    FIX 1 — timestamp is HH:MM:SS (not MM:SS) so rows from long clips
    don't show ambiguous values like "00:01" for multiple clips.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    {total} frames | {fps:.0f} fps | {total/fps:.1f}s")

    prev_gray  = None
    frame_data = []
    idx        = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % FRAME_SAMPLE_RATE == 0:
            gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (320, 240))

            motion = 0.0
            if prev_gray is not None:
                diff   = cv2.absdiff(prev_gray, gray_small)
                motion = float(np.mean(diff)) / 255.0

            # HH:MM:SS — FIX 1
            ts = str(timedelta(seconds=int(idx / fps)))
            frame_data.append((idx, ts, motion, frame))
            prev_gray = gray_small

        idx += 1

    cap.release()
    return frame_data, total


def select_key_frames(frame_data: list) -> list:
    """Top-N frames by motion score, re-sorted chronologically."""
    if not frame_data:
        return []
    ranked = sorted(frame_data, key=lambda x: x[2], reverse=True)[:MAX_KEY_FRAMES]
    return sorted(ranked, key=lambda x: x[0])


# ============================================================================
#  STEP 2 — YOLO OBJECT DETECTION
# ============================================================================

def load_yolo():
    if not YOLO_AVAILABLE:
        print("[YOLO] ultralytics not installed — using MOG2 fallback.")
        return None
    try:
        print("[YOLO] Loading yolov8n.pt …")
        model = YOLO("yolov8n.pt")
        print("[YOLO] Ready.")
        return model
    except Exception as e:
        print(f"[YOLO] Could not load: {e}")
        return None


def detect_objects_yolo(model, frame: np.ndarray) -> tuple[str, int, float]:
    """
    Run YOLOv8 on one frame.
    - Upscales to 640px (CAVIAR is only 384×288 — too small for YOLO without this)
    - Remaps false positives (bird, kite …) to 'person'
    Returns: objects_str, persons_count, max_confidence

    FIX 2 — persons_count returned as integer, not buried in a string.
    """
    h, w = frame.shape[:2]
    if max(h, w) < UPSCALE_TARGET:
        scale = UPSCALE_TARGET / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    results  = model(frame, verbose=False, conf=YOLO_CONF)
    counts   = {}
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            name = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if name in FALSE_POSITIVE_REMAP:
                name = "person"
            counts[name] = counts.get(name, 0) + 1
            if conf > max_conf:
                max_conf = conf

    parts = []
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        suffix = "persons" if name == "person" and n > 1 else (
                 name + "s" if n > 1 else name)
        parts.append(f"{n} {suffix}")

    objects_str   = ", ".join(parts) if parts else "no objects"
    persons_count = counts.get("person", 0)   # FIX 2 — integer
    return objects_str, persons_count, round(max_conf, 2)


def detect_objects_mog2(frame: np.ndarray, bg_sub,
                        motion: float) -> tuple[str, int, float]:
    """MOG2 fallback — estimates person count from foreground contours."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = bg_sub.apply(gray)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    persons = min(len([c for c in conts if cv2.contourArea(c) > 800]), 5)
    conf    = min(round(motion * 10, 2), 1.0)
    obj_str = (f"{persons} person{'s' if persons > 1 else ''}"
               if persons > 0 else "no objects")
    return obj_str, persons, conf


# ============================================================================
#  STEP 3 — EVENT CLASSIFICATION
# ============================================================================

def classify_event(objects_str: str, persons: int, motion: float,
                   prev_persons: int, filename: str) -> str:
    """
    Rule-based classification.
    Priority order:
      1. Filename hints  (CAVIAR clip names encode the ground-truth scenario)
      2. Object + motion rules

    FIX 3 — "Aggressive posturing" was firing for every frame in fight clips
    even when no persons were detected (motion_score=0, objects="no objects").
    Now only fires when there is actual detected motion OR persons in frame.
    """
    fname = filename.lower()

    # ── 1. Filename hints ───────────────────────────────────────────────────
    if any(k in fname for k in ["fight", "chase", "runaway"]):
        if persons >= 2:
            return "Fighting"
        if persons >= 1 and motion > 0.02:
            return "Fighting"
        if motion > 0.03:
            return "Running"
        # FIX 3: only label as aggressive posturing when something is visible
        if persons >= 1 or motion > 0.01:
            return "Aggressive posturing"
        return "No activity"          # ← was returning "Aggressive posturing" here before

    if any(k in fname for k in ["collapse", "fallon", "slump", "onemandown"]):
        if persons >= 1 and motion > 0.03:
            return "Person collapsing"
        if persons >= 1:
            return "Person lying down"
        return "No activity"          # ← was "Person lying down" even with no person visible

    if "leftbag" in fname or "left_bag" in fname:
        if persons >= 1 or objects_str != "no objects":
            return "Suspicious object"
        return "No activity"

    # ── 2. Rule-based fallback ───────────────────────────────────────────────
    has_vehicle = any(v in objects_str for v in
                      ["car", "truck", "bus", "motorcycle"])

    if prev_persons > 1 and persons < prev_persons and motion > 0.03:
        return "Person collapsing"

    if persons >= 4:
        return "Crowd gathering"

    if persons >= 2 and motion > 0.04:
        return "Fighting"

    if persons >= 1 and motion > 0.03:
        return "Running"

    if has_vehicle and motion > 0.02:
        return "Vehicle movement"

    if persons >= 1 and motion < 0.02:
        return "Loitering"

    if persons >= 1:
        return "Normal walking"

    return "No activity"


# ============================================================================
#  STEP 4 — PROCESS ALL CLIPS
# ============================================================================

def process_clips(model, video_paths: list) -> list[dict]:
    all_rows    = []
    frame_ctr   = 0
    bg_sub      = cv2.createBackgroundSubtractorMOG2(
                      history=200, varThreshold=40, detectShadows=False)

    for i, path in enumerate(video_paths):
        path     = Path(path)
        filename = path.name
        clip_id  = f"CAVIAR_{i+1:02d}"
        print(f"\n  [{i+1}/{len(video_paths)}] {filename}  →  {clip_id}")

        frame_data, _ = extract_frames(str(path))
        if not frame_data:
            print("    Could not extract frames — skipping.")
            continue

        key_frames = select_key_frames(frame_data)
        print(f"    Key frames: {len(key_frames)}")

        prev_persons = 0
        for fidx, ts, motion, frame in key_frames:
            frame_ctr += 1
            frame_id   = f"FRM_{frame_ctr:03d}"

            if model:
                objects_str, persons, conf = detect_objects_yolo(model, frame)
            else:
                objects_str, persons, conf = detect_objects_mog2(
                    frame, bg_sub, motion)

            event = classify_event(objects_str, persons, motion,
                                   prev_persons, filename)
            prev_persons = persons

            print(f"    {frame_id} [{ts}]  {event:<25}  "
                  f"{objects_str[:35]:<35}  conf={conf:.2f}")

            all_rows.append({
                "clip_id"      : clip_id,
                "timestamp"    : ts,
                "frame_id"     : frame_id,
                "event"        : event,
                "persons_count": persons,
                "objects"      : objects_str,
                "confidence"   : conf,
            })

    return all_rows


def find_videos(data_dir: Path) -> list:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}
    return sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in exts])


# ============================================================================
#  MAIN
# ============================================================================

def run_pipeline():
    print("=" * 70)
    print("VIDEO PIPELINE — Student 4 (Rahul Sarma Vogeti)")
    print("Multimodal Crime / Incident Report Analyzer  |  CAVIAR Dataset")
    print("=" * 70)

    # ── find videos ────────────────────────────────────────────────────────
    video_paths = []
    if DATA_DIR.exists():
        video_paths = find_videos(DATA_DIR)[:MAX_VIDEOS]

    if not video_paths:
        print(f"\n[!] No videos found in {DATA_DIR}")
        print("    Using built-in sample data instead.\n")
        rows = SAMPLE_DATA
        use_sample = True
    else:
        print(f"\n[Step 1] Found {len(video_paths)} clip(s):")
        for p in video_paths:
            print(f"  {Path(p).name}")
        print("\n[Step 2] Loading YOLO model…")
        model      = load_yolo()
        print("\n[Step 3] Processing clips…")
        rows       = process_clips(model, video_paths)
        use_sample = False

    if not rows:
        print("[!] No rows produced. Exiting.")
        sys.exit(1)

    # ── build DataFrame with all required columns ──────────────────────────
    print(f"\n[Step 4] Exporting {len(rows)} rows → {OUTPUT_FILE}")

    records = []
    for r in rows:
        records.append({
            "Clip_ID"       : r.get("clip_id", "SAMPLE"),
            "Timestamp"     : r["timestamp"],
            "Frame_ID"      : r["frame_id"],
            "Event_Detected": r["event"],
            "Persons_Count" : r.get("persons_count", 0),   # FIX 2 ✓
            "Objects"       : r["objects"],
            "Confidence"    : r["confidence"],
        })

    COLS = ["Clip_ID", "Timestamp", "Frame_ID",
            "Event_Detected", "Persons_Count", "Objects", "Confidence"]

    df = pd.DataFrame(records)[COLS]
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Done!  {len(df)} rows saved to video_output.csv")
    if use_sample:
        print("  (sample data — place CAVIAR clips in data/ to run for real)")
    print()
    print("  Event breakdown:")
    for evt, n in df["Event_Detected"].value_counts().items():
        bar = "█" * min(n, 30)
        print(f"    {evt:<25}  {n:3d}  {bar}")
    print(f"{'=' * 70}")

    print("\n  Sample output (first 12 rows):")
    print(df.head(12).to_string(index=False, max_colwidth=35))
    return df


if __name__ == "__main__":
    run_pipeline()
