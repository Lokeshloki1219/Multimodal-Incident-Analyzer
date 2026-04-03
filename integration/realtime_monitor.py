"""
Real-Time Processing Monitor — Multimodal Crime / Incident Report Analyzer

Watches the /new_data folder for incoming files and automatically routes them
to the appropriate pipeline based on file extension:

    .wav, .mp3, .flac   →  Audio pipeline  →  audio_output.csv
    .pdf                 →  PDF pipeline    →  pdf_output.csv
    .jpg, .png, .bmp     →  Image pipeline  →  image_output.csv
    .mp4, .avi, .mpg     →  Video pipeline  →  video_output.csv
    .csv, .txt, .json    →  Text pipeline   →  text_output.csv

After processing, the integration merge is re-run to update final_incidents.csv
with the new incident. The LLM summarizer generates a summary for only the
new row (not the entire dataset). The Streamlit dashboard auto-refreshes to
pick up the updated CSV.

Usage:
    Terminal 1:  python realtime_monitor.py
    Terminal 2:  streamlit run dashboard.py

    Then drag/drop a file into the new_data/ folder and watch the dashboard update.

Requirements:
    pip install watchdog
"""

import os
import sys
import time
import shutil
import threading
import logging
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data")
FINAL_CSV = os.path.join(SCRIPT_DIR, "final_incidents.csv")
LOCK_FILE = os.path.join(SCRIPT_DIR, ".processing.lock")

# File extension → modality mapping
EXTENSION_MAP = {
    # Audio
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    # PDF
    ".pdf": "pdf",
    # Image
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image",
    # Video
    ".mp4": "video", ".avi": "video", ".mpg": "video",
    ".mpeg": "video", ".mov": "video", ".mkv": "video",
    # Text
    ".csv": "text", ".txt": "text", ".json": "text",
}

# Modality CSV paths (same as integrate.py)
CSV_PATHS = {
    "audio": os.path.join(BASE_DIR, "audio", "audio_output.csv"),
    "pdf": os.path.join(BASE_DIR, "pdf", "pdf_output.csv"),
    "image": os.path.join(BASE_DIR, "images", "image_output.csv"),
    "video": os.path.join(BASE_DIR, "video", "video_output.csv"),
    "text": os.path.join(BASE_DIR, "text", "text_output.csv"),
}

# Processing archive — processed files are moved here
ARCHIVE_DIR = os.path.join(NEW_DATA_DIR, "_processed")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("RealTimeMonitor")


# ============================================================================
# LOCK MECHANISM — prevents race conditions with simultaneous files
# ============================================================================

class ProcessingLock:
    """Simple file-based lock to prevent concurrent processing."""

    def __init__(self, lock_path):
        self.lock_path = lock_path

    def acquire(self, timeout=30):
        """Wait up to 'timeout' seconds to acquire the lock."""
        start = time.time()
        while os.path.exists(self.lock_path):
            if time.time() - start > timeout:
                log.warning("Lock timeout — forcing lock release")
                self.release()
                break
            time.sleep(0.5)
        with open(self.lock_path, 'w') as f:
            f.write(str(os.getpid()))
        return True

    def release(self):
        """Release the lock."""
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


lock = ProcessingLock(LOCK_FILE)


# ============================================================================
# PER-MODALITY PROCESSING — lightweight single-file processors
# ============================================================================

def process_audio_file(file_path):
    """
    Process a single audio file through the audio pipeline.
    Returns a dict with audio output columns.
    """
    log.info(f"  [Audio] Processing: {os.path.basename(file_path)}")
    try:
        # Add audio pipeline to path
        audio_dir = os.path.join(BASE_DIR, "audio")
        if audio_dir not in sys.path:
            sys.path.insert(0, audio_dir)

        import whisper
        import spacy

        # Load models (cached after first call)
        if not hasattr(process_audio_file, '_whisper_model'):
            log.info("  [Audio] Loading Whisper model (first time)...")
            process_audio_file._whisper_model = whisper.load_model("base")
        if not hasattr(process_audio_file, '_nlp'):
            process_audio_file._nlp = spacy.load("en_core_web_sm")

        model = process_audio_file._whisper_model
        nlp = process_audio_file._nlp

        # Transcribe
        result = model.transcribe(str(file_path), language="en")
        transcript = result["text"].strip()
        log.info(f"  [Audio] Transcript: {transcript[:80]}...")

        # Extract event using spaCy NER
        doc = nlp(transcript)
        events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]
        locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]

        # Keyword-based event extraction (fallback)
        event_keywords = {
            "fire": "Fire", "shooting": "Shooting", "robbery": "Robbery",
            "accident": "Accident", "fight": "Fight", "assault": "Assault",
            "break": "Break-in", "theft": "Theft", "medical": "Medical Emergency",
            "stabbing": "Stabbing", "explosion": "Explosion",
        }
        extracted_event = "General Emergency"
        transcript_lower = transcript.lower()
        for keyword, event_label in event_keywords.items():
            if keyword in transcript_lower:
                extracted_event = event_label
                break
        if events:
            extracted_event = events[0]

        location = locations[0] if locations else "Unknown"

        # Sentiment (simple keyword approach for speed)
        neg_words = ["help", "emergency", "fire", "hurt", "dead", "shot",
                     "bleeding", "trapped", "scream", "attack"]
        neg_count = sum(1 for w in neg_words if w in transcript_lower)
        sentiment = "Negative" if neg_count >= 2 else ("Neutral" if neg_count == 1 else "Positive")

        # Urgency score
        from audio_pipeline import URGENCY_KEYWORDS
        urgency = sum(w for kw, w in URGENCY_KEYWORDS.items() if kw in transcript_lower)
        urgency_score = round(min(urgency, 1.0), 2)

        return {
            "Transcript": transcript[:500],
            "Extracted_Event": extracted_event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency_score,
        }

    except Exception as e:
        log.error(f"  [Audio] Error: {e}")
        return {
            "Transcript": f"[Processing error: {e}]",
            "Extracted_Event": "Unknown",
            "Location": "Unknown",
            "Sentiment": "Neutral",
            "Urgency_Score": 0.5,
        }


def process_image_file(file_path):
    """
    Process a single image file through the image pipeline.
    Returns a dict with image output columns.
    """
    log.info(f"  [Image] Processing: {os.path.basename(file_path)}")
    try:
        images_dir = os.path.join(BASE_DIR, "images")
        if images_dir not in sys.path:
            sys.path.insert(0, images_dir)

        from ultralytics import YOLO

        # Load COCO model (cached)
        if not hasattr(process_image_file, '_coco_model'):
            log.info("  [Image] Loading YOLOv8n model (first time)...")
            process_image_file._coco_model = YOLO("yolov8n.pt")

        model = process_image_file._coco_model
        results = model(str(file_path), verbose=False)

        objects = {}
        max_conf = 0.0
        for r in results:
            for box in r.boxes:
                name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                objects[name] = objects.get(name, 0) + 1
                max_conf = max(max_conf, conf)

        # Format objects string
        parts = []
        for name, count in sorted(objects.items(), key=lambda x: -x[1]):
            if name == "person" and count > 1:
                parts.append(f"{count} persons")
            elif count > 1:
                parts.append(f"{count} {name}s")
            else:
                parts.append(name)
        objects_str = ", ".join(parts) if parts else "none detected"

        # Scene classification (keyword-based for speed)
        scene = "General Scene"
        obj_set = set(objects.keys())
        if obj_set & {"fire", "smoke"}:
            scene = "Fire Scene"
        elif obj_set & {"car", "truck"} and "person" in obj_set:
            scene = "Vehicle Accident"
        elif "person" in obj_set and objects.get("person", 0) >= 3:
            scene = "Public Disturbance"
        elif "person" in obj_set:
            scene = "Surveillance Scene"

        # OCR
        text_extracted = ""
        try:
            import pytesseract
            from PIL import Image
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            img = Image.open(file_path)
            text_extracted = pytesseract.image_to_string(img).strip()[:200]
        except Exception:
            pass

        # Bounding box description
        bbox_parts = [f"{count} {name}" for name, count in objects.items()]
        bbox_desc = ", ".join(bbox_parts) if bbox_parts else "none"

        return {
            "Scene_Type": scene,
            "Objects_Detected": objects_str,
            "Bounding_Boxes": bbox_desc,
            "Text_Extracted": text_extracted,
            "Confidence_Score": round(max_conf, 2) if max_conf > 0 else 0.50,
        }

    except Exception as e:
        log.error(f"  [Image] Error: {e}")
        return {
            "Scene_Type": "Unknown",
            "Objects_Detected": "processing error",
            "Bounding_Boxes": "none",
            "Text_Extracted": "",
            "Confidence_Score": 0.50,
        }


def process_pdf_file(file_path):
    """
    Process a single PDF file through a simplified PDF pipeline.
    Returns a dict with PDF output columns.
    """
    log.info(f"  [PDF] Processing: {os.path.basename(file_path)}")
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        text_lower = full_text.lower()

        # Classify incident type
        incident_type = "General Report"
        if "fire" in text_lower or "arson" in text_lower:
            incident_type = "Arson / Fire Investigation"
        elif "narcotics" in text_lower or "drug" in text_lower:
            incident_type = "Narcotics Investigation"
        elif "theft" in text_lower or "robbery" in text_lower:
            incident_type = "Theft / Robbery"
        elif "assault" in text_lower:
            incident_type = "Assault"

        # Doc type
        doc_type = "Report"
        if "training" in text_lower:
            doc_type = "Training Proposal"
        elif "procedure" in text_lower or "sop" in text_lower:
            doc_type = "SOP"
        elif "inventory" in text_lower:
            doc_type = "Inventory Report"

        # Extract date
        import re
        date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', full_text)
        date = date_match.group(0) if date_match else "Unknown"

        # Summary
        sentences = [s.strip() for s in full_text[:1000].split('.') if len(s.strip()) > 20]
        summary = sentences[0] + "." if sentences else "Document processed"

        return {
            "Department": "Unknown",
            "Incident_Type": incident_type,
            "Doc_Type": doc_type,
            "Date": date,
            "Location": "Unknown",
            "Program": "General Operations",
            "Officer": "Unknown",
            "Summary": summary[:200],
            "Key_Detail": summary.split(".")[0][:100] if summary else "Document section",
        }

    except Exception as e:
        log.error(f"  [PDF] Error: {e}")
        return {
            "Department": "Unknown", "Incident_Type": "Unknown",
            "Doc_Type": "Report", "Date": "Unknown", "Location": "Unknown",
            "Program": "Unknown", "Officer": "Unknown",
            "Summary": f"Processing error: {e}", "Key_Detail": "Error",
        }


def process_video_file(file_path):
    """
    Process a single video file — extracts key frames and runs YOLO.
    Returns a dict with video output columns.
    """
    log.info(f"  [Video] Processing: {os.path.basename(file_path)}")
    try:
        import cv2
        from ultralytics import YOLO

        if not hasattr(process_video_file, '_model'):
            process_video_file._model = YOLO("yolov8n.pt")
        model = process_video_file._model

        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample middle frame for quick analysis
        mid_frame_idx = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video frame")

        # YOLO detection on sampled frame
        results = model(frame, verbose=False)
        objects = {}
        max_conf = 0.0
        for r in results:
            for box in r.boxes:
                name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                objects[name] = objects.get(name, 0) + 1
                max_conf = max(max_conf, conf)

        parts = []
        for name, count in sorted(objects.items(), key=lambda x: -x[1]):
            if name == "person" and count > 1:
                parts.append(f"{count} persons")
            elif count > 1:
                parts.append(f"{count} {name}s")
            else:
                parts.append(f"1 {name}")
        objects_str = ", ".join(parts) if parts else "no objects"

        # Person count
        persons_count = objects.get("person", 0)

        # Event classification
        event = "Normal activity"
        if persons_count >= 3:
            event = "Group gathering"
        elif persons_count >= 2:
            event = "Multiple persons"
        elif any(v in objects for v in ["car", "truck"]):
            event = "Vehicle movement"
        elif persons_count == 1:
            event = "Normal walking"

        timestamp = f"{int(mid_frame_idx / fps // 60):02d}:{int(mid_frame_idx / fps % 60):02d}"

        return {
            "Timestamp": timestamp,
            "Frame_ID": f"FRM_RT_{int(time.time()) % 10000:04d}",
            "Event_Detected": event,
            "Persons_Count": persons_count,
            "Objects": objects_str,
            "Confidence": round(max_conf, 2) if max_conf > 0 else 0.50,
        }

    except Exception as e:
        log.error(f"  [Video] Error: {e}")
        return {
            "Timestamp": "00:00", "Frame_ID": "FRM_ERR",
            "Event_Detected": "Unknown", "Persons_Count": 0,
            "Objects": "processing error", "Confidence": 0.50,
        }


def process_text_file(file_path):
    """
    Process a single text/CSV/JSON file through the text pipeline.
    Returns a dict with text output columns.
    """
    log.info(f"  [Text] Processing: {os.path.basename(file_path)}")
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".json":
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get("text", str(data))
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            text = " ".join(df.iloc[0].astype(str).tolist()) if len(df) > 0 else ""
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        text = text[:2000]
        text_lower = text.lower()

        # Zero-shot topic classification (keyword fallback for speed)
        topic = "General Crime"
        topic_map = {
            "Murder / Homicide": ["murder", "killed", "homicide", "dead"],
            "Robbery / Theft": ["robbery", "theft", "stole", "stolen", "robbed"],
            "Assault / Violence": ["assault", "attack", "beaten", "stabbed"],
            "Arson / Fire": ["fire", "arson", "blaze", "burned"],
            "Drug Crime": ["drug", "narcotics", "cocaine", "heroin"],
            "Shooting": ["shooting", "shot", "gunfire", "gunshot"],
            "Traffic Accident": ["accident", "crash", "collision", "hit-and-run"],
        }
        for label, keywords in topic_map.items():
            if any(kw in text_lower for kw in keywords):
                topic = label
                break

        # Sentiment
        neg_words = ["killed", "dead", "shot", "fire", "robbery", "assault",
                     "arrested", "stabbed", "crash", "attack"]
        neg_count = sum(1 for w in neg_words if w in text_lower)
        sentiment = "Negative" if neg_count >= 2 else "Neutral"

        # Severity
        high_kw = ["killed", "dead", "murder", "shot", "shooting", "fatal"]
        severity = "High" if any(kw in text_lower for kw in high_kw) else (
            "Medium" if neg_count >= 2 else "Low")

        return {
            "Source": "RealTime",
            "Raw_Text": text[:300],
            "Crime_Type": topic,
            "Location_Entity": "Unknown",
            "Sentiment": sentiment,
            "Entities": "Real-time processed",
            "Topic": topic,
            "Severity_Label": severity,
        }

    except Exception as e:
        log.error(f"  [Text] Error: {e}")
        return {
            "Source": "RealTime", "Raw_Text": f"Error: {e}",
            "Crime_Type": "Unknown", "Location_Entity": "Unknown",
            "Sentiment": "Neutral", "Entities": "Error",
            "Topic": "Unknown", "Severity_Label": "Low",
        }


# ============================================================================
# APPEND TO MODALITY CSV
# ============================================================================

def append_to_modality_csv(modality, result):
    """Append a new row to the appropriate modality CSV with the next ID."""
    csv_path = CSV_PATHS[modality]

    # Read existing CSV to determine next ID
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        next_idx = len(existing) + 1
    else:
        existing = pd.DataFrame()
        next_idx = 1

    # Generate ID based on modality
    id_col_map = {
        "audio": ("Call_ID", f"CALL_{next_idx:03d}"),
        "pdf": ("Report_ID", f"RPT_{next_idx:03d}"),
        "image": ("Image_ID", f"IMG_{next_idx:03d}"),
        "video": ("Clip_ID", f"CAVIAR_{next_idx:02d}"),
        "text": ("Text_ID", f"TXT_{next_idx:03d}"),
    }
    id_col, id_val = id_col_map[modality]
    result[id_col] = id_val

    # Append to CSV
    new_row = pd.DataFrame([result])
    if not existing.empty:
        # Ensure column order matches existing
        for col in existing.columns:
            if col not in new_row.columns:
                new_row[col] = "N/A"
        new_row = new_row[existing.columns]
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(csv_path, index=False, encoding='utf-8')
    log.info(f"  → Appended {id_val} to {os.path.basename(csv_path)} "
             f"(now {len(updated)} rows)")

    return id_val


# ============================================================================
# INCREMENTAL INTEGRATION — append new incident to final_incidents.csv
# ============================================================================

def append_to_final_incidents(modality, result, modality_id):
    """
    Append a single new incident to final_incidents.csv.
    Only processes the new row — does NOT re-run the full integration.
    """
    import pandas as pd

    # Missing value defaults
    MISSING = {
        "audio": "No 911 call recorded",
        "pdf": "No police report filed",
        "image": "No scene photograph available",
        "video": "No CCTV footage available",
        "text": "No media report found",
    }

    # Load existing final CSV
    if os.path.exists(FINAL_CSV):
        final_df = pd.read_csv(FINAL_CSV)
        next_inc = len(final_df) + 1
    else:
        final_df = pd.DataFrame()
        next_inc = 1

    incident_id = f"INC_{next_inc:03d}"

    # Build the new incident row
    new_incident = {
        "Incident_ID": incident_id,
        "Audio_Event": MISSING["audio"],
        "PDF_Doc_Type": MISSING["pdf"],
        "Image_Objects": MISSING["image"],
        "Video_Event": MISSING["video"],
        "Text_Crime_Type": MISSING["text"],
        "Severity": "Unknown",
        "Sources_Available": "",
        "Modality_Count": 1,
    }

    # Fill in the modality-specific data
    if modality == "audio":
        new_incident["Audio_Event"] = result.get("Extracted_Event", "Unknown")
        new_incident["Sources_Available"] = "Audio"
        # Compute severity from urgency
        urgency = float(result.get("Urgency_Score", 0.5))
        new_incident["Severity"] = "High" if urgency > 0.7 else (
            "Medium" if urgency > 0.4 else "Low")

    elif modality == "pdf":
        new_incident["PDF_Doc_Type"] = result.get("Doc_Type", "Report")
        new_incident["Sources_Available"] = "PDF"
        new_incident["Severity"] = "Medium"

    elif modality == "image":
        conf = float(result.get("Confidence_Score", 0.5))
        objects = result.get("Objects_Detected", "none")
        new_incident["Image_Objects"] = f"{objects} ({conf:.2f})"
        new_incident["Sources_Available"] = "Image"
        new_incident["Severity"] = "High" if conf > 0.7 else (
            "Medium" if conf > 0.4 else "Low")

    elif modality == "video":
        new_incident["Video_Event"] = result.get("Event_Detected", "Unknown")
        new_incident["Sources_Available"] = "Video"
        conf = float(result.get("Confidence", 0.5))
        new_incident["Severity"] = "High" if conf > 0.7 else (
            "Medium" if conf > 0.4 else "Low")

    elif modality == "text":
        new_incident["Text_Crime_Type"] = result.get("Crime_Type", "Unknown")
        new_incident["Sources_Available"] = "Text"
        sev_label = result.get("Severity_Label", "Low")
        new_incident["Severity"] = sev_label

    # Append to final CSV
    new_row = pd.DataFrame([new_incident])

    # Ensure column order
    final_cols = [
        "Incident_ID", "Audio_Event", "PDF_Doc_Type", "Image_Objects",
        "Video_Event", "Text_Crime_Type", "Severity",
        "Sources_Available", "Modality_Count"
    ]

    # Preserve AI_Summary column if it exists
    if not final_df.empty and "AI_Summary" in final_df.columns:
        final_cols.append("AI_Summary")
        new_row["AI_Summary"] = ""

    for col in final_cols:
        if col not in new_row.columns:
            new_row[col] = "N/A"
    new_row = new_row[final_cols]

    if not final_df.empty:
        # Ensure same columns
        for col in final_cols:
            if col not in final_df.columns:
                final_df[col] = "N/A"
        final_df = final_df[final_cols]
        updated = pd.concat([final_df, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(FINAL_CSV, index=False, encoding='utf-8')
    log.info(f"  → Created {incident_id} in final_incidents.csv "
             f"(now {len(updated)} incidents)")

    return incident_id, new_incident


# ============================================================================
# LLM SUMMARY FOR NEW ROW ONLY
# ============================================================================

def summarize_single_incident(incident_id):
    """
    Run Flan-T5 on just the new incident row — avoids reprocessing all rows.
    """
    try:
        # Import from summarizer module
        sys.path.insert(0, SCRIPT_DIR)
        from summarizer import load_model, build_prompt, generate_summary

        df = pd.read_csv(FINAL_CSV)
        row = df[df["Incident_ID"] == incident_id].iloc[0]

        # Load model (cached across calls)
        if not hasattr(summarize_single_incident, '_model'):
            log.info("  [LLM] Loading Flan-T5 model (first time)...")
            model, tokenizer, device = load_model()
            summarize_single_incident._model = model
            summarize_single_incident._tokenizer = tokenizer
            summarize_single_incident._device = device

        model = summarize_single_incident._model
        tokenizer = summarize_single_incident._tokenizer
        device = summarize_single_incident._device

        prompt = build_prompt(row)
        summary = generate_summary(model, tokenizer, device, prompt)

        # Update only this row's AI_Summary
        if "AI_Summary" not in df.columns:
            df["AI_Summary"] = ""
        df.loc[df["Incident_ID"] == incident_id, "AI_Summary"] = summary
        df.to_csv(FINAL_CSV, index=False, encoding='utf-8')

        log.info(f"  [LLM] Summary: {summary[:80]}...")
        return summary

    except ImportError:
        log.warning("  [LLM] transformers not installed — skipping summarization")
        return ""
    except Exception as e:
        log.warning(f"  [LLM] Summarization error: {e}")
        return ""


# ============================================================================
# MAIN FILE PROCESSING HANDLER
# ============================================================================

def process_new_file(file_path):
    """
    Main entry point for processing a new file:
    1. Detect modality from extension
    2. Run the appropriate pipeline
    3. Append result to modality CSV
    4. Append new incident to final_incidents.csv
    5. Generate AI summary for the new incident
    6. Archive the processed file
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in EXTENSION_MAP:
        log.warning(f"Unknown file type: {ext} — skipping {filename}")
        return

    modality = EXTENSION_MAP[ext]

    log.info("=" * 60)
    log.info(f"NEW FILE DETECTED: {filename}")
    log.info(f"Modality: {modality.upper()} | Extension: {ext}")
    log.info("=" * 60)

    with lock:
        start_time = time.time()

        # Step 1: Process through pipeline
        log.info("[Step 1/4] Running pipeline...")
        processors = {
            "audio": process_audio_file,
            "pdf": process_pdf_file,
            "image": process_image_file,
            "video": process_video_file,
            "text": process_text_file,
        }
        result = processors[modality](file_path)

        # Step 2: Append to modality CSV
        log.info("[Step 2/4] Appending to modality CSV...")
        modality_id = append_to_modality_csv(modality, result)

        # Step 3: Append to final_incidents.csv
        log.info("[Step 3/4] Creating incident in final_incidents.csv...")
        incident_id, incident = append_to_final_incidents(
            modality, result, modality_id)

        # Step 4: Generate AI summary (new row only)
        log.info("[Step 4/4] Generating AI summary...")
        summarize_single_incident(incident_id)

        elapsed = time.time() - start_time
        log.info(f"\n{'=' * 60}")
        log.info(f"PROCESSING COMPLETE in {elapsed:.1f}s")
        log.info(f"  File:      {filename}")
        log.info(f"  Modality:  {modality.upper()}")
        log.info(f"  Entry:     {modality_id}")
        log.info(f"  Incident:  {incident_id}")
        log.info(f"  Severity:  {incident['Severity']}")
        log.info(f"{'=' * 60}\n")

    # Archive processed file
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archive_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    archive_path = os.path.join(ARCHIVE_DIR, archive_name)
    try:
        shutil.move(file_path, archive_path)
        log.info(f"  Archived → {archive_name}")
    except Exception as e:
        log.warning(f"  Could not archive: {e}")


# ============================================================================
# WATCHDOG FILE SYSTEM MONITOR
# ============================================================================

def start_watchdog_monitor():
    """Start the watchdog observer to monitor the new_data folder."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log.error("watchdog not installed! Run: pip install watchdog")
        sys.exit(1)

    class NewFileHandler(FileSystemEventHandler):
        """Handle new files dropped into the new_data folder."""

        def __init__(self):
            self.debounce = {}  # path -> timestamp (prevent double events)

        def on_created(self, event):
            if event.is_directory:
                return

            file_path = event.src_path
            filename = os.path.basename(file_path)

            # Skip hidden files, temp files, processed archive
            if filename.startswith('.') or filename.startswith('~'):
                return
            if '_processed' in file_path:
                return

            # Debounce — watchdog can fire multiple events for one file
            now = time.time()
            if file_path in self.debounce and now - self.debounce[file_path] < 3:
                return
            self.debounce[file_path] = now

            # Wait a moment for file to finish writing
            time.sleep(1.0)

            # Verify file still exists and is complete
            if not os.path.exists(file_path):
                return
            if os.path.getsize(file_path) == 0:
                log.warning(f"  Empty file: {filename} — waiting...")
                time.sleep(2.0)
                if os.path.getsize(file_path) == 0:
                    return

            # Process in a separate thread to avoid blocking the observer
            thread = threading.Thread(
                target=process_new_file,
                args=(file_path,),
                daemon=True
            )
            thread.start()

    # Create new_data folder if it doesn't exist
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    handler = NewFileHandler()
    observer = Observer()
    observer.schedule(handler, NEW_DATA_DIR, recursive=False)
    observer.start()

    return observer


# ============================================================================
# MAIN
# ============================================================================

import pandas as pd  # needed at module level for append functions

def main():
    print("=" * 65)
    print("  REAL-TIME MONITOR — Multimodal Crime / Incident Report Analyzer")
    print("=" * 65)
    print()
    print(f"  Watching folder:  {NEW_DATA_DIR}")
    print(f"  Final CSV:        {FINAL_CSV}")
    print()
    print("  Supported file types:")
    print("    Audio:  .wav, .mp3, .flac")
    print("    PDF:    .pdf")
    print("    Image:  .jpg, .png, .bmp")
    print("    Video:  .mp4, .avi, .mpg")
    print("    Text:   .csv, .txt, .json")
    print()
    print("  Drop a file into the new_data/ folder to process it!")
    print("  Press Ctrl+C to stop.")
    print("=" * 65)

    # Ensure new_data directory exists
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    # Check if final_incidents.csv exists
    if os.path.exists(FINAL_CSV):
        df = pd.read_csv(FINAL_CSV)
        print(f"\n  Current incidents: {len(df)}")
    else:
        print("\n  [!] final_incidents.csv not found — run integrate.py first")
        print("      New incidents will still be created from scratch.")

    # Start the watchdog observer
    observer = start_watchdog_monitor()
    print(f"\n  [✓] Monitor started. Waiting for new files...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n  [!] Stopping monitor...")
        observer.stop()

    observer.join()
    print("  Monitor stopped. Goodbye!\n")


if __name__ == "__main__":
    main()
