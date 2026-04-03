"""
Integration Pipeline — Final Team Effort (Stage 4)

Multimodal Crime / Incident Report Analyzer

Strategy:
    Combines all five modality CSVs into a single unified incident dataset.
    Each row represents one incident with data pulled from audio, PDF, image,
    video, and text pipelines. The merge is sequential — incidents are assigned
    INC_001 through INC_198, with varying levels of data completeness based on
    the different row counts per modality.

    - INC_001 to INC_045:  All 5 modalities (Audio + PDF + Image + Video + Text)
    - INC_046 to INC_100:  4 modalities (Audio + Image + Video + Text)
    - INC_101 to INC_113:  3 modalities (Audio + Video + Text)
    - INC_114 to INC_198:  2 modalities (Audio + Video)
    - Beyond INC_198:      Dropped (single-modality = no multimodal integration)

Missing Values:
    Context-aware descriptive text instead of NaN:
    - No audio: "No 911 call recorded"
    - No PDF:   "No police report filed"
    - No image: "No scene photograph available"
    - No video: "No CCTV footage available"
    - No text:  "No media report found"

Severity:
    Adaptive severity from available signals:
    - Audio: Urgency_Score (0-1)
    - Image: Confidence_Score (0-1)
    - Text:  Severity_Label (High=1.0, Medium=0.5, Low=0.2)
    Average of available signals → High (>0.7), Medium (>0.4), Low (<=0.4)

Output:
    final_incidents.csv with columns:
    Incident_ID, Audio_Event, PDF_Doc_Type, Image_Objects, Video_Event,
    Text_Crime_Type, Severity, Sources_Available, Modality_Count
"""

import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "final_incidents.csv")

CSV_PATHS = {
    "audio": os.path.join(BASE_DIR, "audio", "audio_output.csv"),
    "pdf": os.path.join(BASE_DIR, "pdf", "pdf_output.csv"),
    "image": os.path.join(BASE_DIR, "images", "image_output.csv"),
    "video": os.path.join(BASE_DIR, "video", "video_output.csv"),
    "text": os.path.join(BASE_DIR, "text", "text_output.csv"),
}

# Context-aware missing value descriptions
MISSING_VALUES = {
    "audio": "No 911 call recorded",
    "pdf": "No police report filed",
    "image": "No scene photograph available",
    "video": "No CCTV footage available",
    "text": "No media report found",
}

# Severity label to numeric mapping
SEVERITY_MAP = {
    "High": 1.0,
    "Medium": 0.5,
    "Low": 0.2,
}


# ============================================================================
# STEP 1: LOAD ALL MODALITY CSVS
# ============================================================================

def load_all_csvs():
    """Load all five modality CSVs and return as dict of DataFrames."""
    data = {}
    print("\n  Loading modality outputs...")

    for modality, path in CSV_PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='utf-8')
            data[modality] = df
            print(f"    [OK] {modality.upper():6s}: {len(df):4d} rows  "
                  f"cols={list(df.columns)[:4]}...")
        else:
            print(f"    [--] {modality.upper():6s}: FILE NOT FOUND at {path}")
            data[modality] = pd.DataFrame()

    return data


# ============================================================================
# STEP 2: SEQUENTIAL MERGE
# ============================================================================

def build_sequential_merge(data):
    """
    Build the merged incident dataset by sequential alignment.

    Uses the second-largest modality (Video=198) as the row count ceiling,
    since single-modality rows don't demonstrate multimodal integration.

    For each row index i:
    - Audio[i] if i < len(audio), else missing
    - PDF[i] if i < len(pdf), else missing
    - Image[i] if i < len(image), else missing
    - Video[i] if i < len(video), else missing
    - Text[i] if i < len(text), else missing
    """
    # Determine row counts
    counts = {m: len(df) for m, df in data.items() if not df.empty}
    print(f"\n  Row counts: {counts}")

    # Sort by count descending to find the merge ceiling
    sorted_counts = sorted(counts.values(), reverse=True)

    # Use the second-largest count as ceiling (at least 2 modalities needed)
    # If we only have 1 modality, use that count
    if len(sorted_counts) >= 2:
        merge_ceiling = sorted_counts[1]  # Second largest = max rows with >=2 modalities
    else:
        merge_ceiling = sorted_counts[0] if sorted_counts else 0

    # But we want at least 2 modalities per row, so find the actual ceiling
    # by checking how many modalities have data at each index
    max_rows = max(counts.values()) if counts else 0
    actual_ceiling = 0
    for i in range(max_rows):
        modality_count = sum(1 for m, df in data.items()
                            if not df.empty and i < len(df))
        if modality_count >= 2:
            actual_ceiling = i + 1

    print(f"  Merge ceiling: {actual_ceiling} rows (at least 2 modalities per row)")

    # Build rows
    incidents = []
    for i in range(actual_ceiling):
        row = {"Incident_ID": f"INC_{i + 1:03d}"}
        sources = []

        # --- AUDIO ---
        if not data["audio"].empty and i < len(data["audio"]):
            audio_row = data["audio"].iloc[i]
            row["Audio_Event"] = str(audio_row.get("Extracted_Event", "N/A"))
            row["_audio_urgency"] = float(audio_row.get("Urgency_Score", 0.0))
            sources.append("Audio")
        else:
            row["Audio_Event"] = MISSING_VALUES["audio"]
            row["_audio_urgency"] = None

        # --- PDF ---
        if not data["pdf"].empty and i < len(data["pdf"]):
            pdf_row = data["pdf"].iloc[i]
            row["PDF_Doc_Type"] = str(pdf_row.get("Doc_Type",
                                      pdf_row.get("Incident_Type", "N/A")))
            sources.append("PDF")
        else:
            row["PDF_Doc_Type"] = MISSING_VALUES["pdf"]

        # --- IMAGE ---
        if not data["image"].empty and i < len(data["image"]):
            img_row = data["image"].iloc[i]
            objects = str(img_row.get("Objects_Detected", "none"))
            conf = float(img_row.get("Confidence_Score", 0.0))
            row["Image_Objects"] = f"{objects} ({conf:.2f})" if objects != "none detected" else f"none ({conf:.2f})"
            row["_image_confidence"] = conf
            sources.append("Image")
        else:
            row["Image_Objects"] = MISSING_VALUES["image"]
            row["_image_confidence"] = None

        # --- VIDEO ---
        if not data["video"].empty and i < len(data["video"]):
            video_row = data["video"].iloc[i]
            row["Video_Event"] = str(video_row.get("Event_Detected", "N/A"))
            sources.append("Video")
        else:
            row["Video_Event"] = MISSING_VALUES["video"]

        # --- TEXT ---
        if not data["text"].empty and i < len(data["text"]):
            text_row = data["text"].iloc[i]
            row["Text_Crime_Type"] = str(text_row.get("Crime_Type",
                                         text_row.get("Topic", "N/A")))
            severity_label = str(text_row.get("Severity_Label", "Low"))
            row["_text_severity"] = SEVERITY_MAP.get(severity_label, 0.2)
            sources.append("Text")
        else:
            row["Text_Crime_Type"] = MISSING_VALUES["text"]
            row["_text_severity"] = None

        # Metadata
        row["Sources_Available"] = " + ".join(sources)
        row["Modality_Count"] = len(sources)

        incidents.append(row)

    return incidents


# ============================================================================
# STEP 3: SEVERITY CLASSIFICATION
# ============================================================================

def compute_severity(incident):
    """
    Compute severity from available modality signals.

    Signals used:
    - Audio: Urgency_Score (0-1)
    - Image: Confidence_Score (0-1)
    - Text:  Severity_Label mapped to numeric (High=1.0, Medium=0.5, Low=0.2)

    Average of available signals:
    - > 0.7 → High
    - > 0.4 → Medium
    - <= 0.4 → Low
    - No signals → Unknown
    """
    signals = []

    audio_urgency = incident.get("_audio_urgency")
    if audio_urgency is not None:
        signals.append(audio_urgency)

    image_confidence = incident.get("_image_confidence")
    if image_confidence is not None:
        signals.append(image_confidence)

    text_severity = incident.get("_text_severity")
    if text_severity is not None:
        signals.append(text_severity)

    if not signals:
        return "Unknown"

    avg = sum(signals) / len(signals)

    if avg > 0.7:
        return "High"
    elif avg > 0.4:
        return "Medium"
    else:
        return "Low"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_integration():
    """Execute the sequential merge integration pipeline."""
    print("=" * 70)
    print("INTEGRATION PIPELINE — Sequential Multimodal Merge")
    print("Multimodal Crime / Incident Report Analyzer")
    print("=" * 70)

    # Step 1: Load all CSVs
    print("\n[Step 1] Loading all modality outputs...")
    data = load_all_csvs()

    # Step 2: Sequential merge
    print("\n[Step 2] Building sequential merge...")
    incidents = build_sequential_merge(data)
    print(f"\n  Total incidents created: {len(incidents)}")

    if not incidents:
        print("[Integration] ERROR: No incidents created. Exiting.")
        sys.exit(1)

    # Step 3: Compute severity
    print("\n[Step 3] Computing adaptive severity classification...")
    for inc in incidents:
        inc["Severity"] = compute_severity(inc)

    df = pd.DataFrame(incidents)

    # Show severity distribution
    severity_counts = df["Severity"].value_counts()
    for sev in ["High", "Medium", "Low", "Unknown"]:
        cnt = severity_counts.get(sev, 0)
        if cnt > 0:
            print(f"    {sev}: {cnt} incidents")

    # Show modality count distribution
    print("\n  Modality coverage:")
    modality_counts = df["Modality_Count"].value_counts().sort_index(ascending=False)
    for count, num_incidents in modality_counts.items():
        print(f"    {count} modalities: {num_incidents} incidents")

    # Step 4: Build final output
    print("\n[Step 4] Building final output...")

    # Assignment-required columns + metadata
    final_cols = [
        "Incident_ID", "Audio_Event", "PDF_Doc_Type", "Image_Objects",
        "Video_Event", "Text_Crime_Type", "Severity",
        "Sources_Available", "Modality_Count"
    ]

    # Drop internal scoring columns (prefixed with _)
    for col in list(df.columns):
        if col.startswith("_"):
            df = df.drop(columns=[col])

    # Ensure all columns exist
    for col in final_cols:
        if col not in df.columns:
            df[col] = "N/A"

    final_df = df[final_cols]

    # Step 5: Export
    print(f"\n[Step 5] Exporting to '{OUTPUT_FILE}'...")
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved: {OUTPUT_FILE} ({len(final_df)} rows, {len(final_cols)} columns)")

    print(f"\n{'=' * 70}")
    print("INTEGRATION COMPLETE!")
    print(f"  Total incidents: {len(final_df)}")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"{'=' * 70}")

    # Preview
    print("\n--- Final Output Preview (first 10) ---")
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 220)
    print(final_df.head(10).to_string(index=False))

    # Incident range summary
    print("\n--- Incident Coverage Summary ---")
    for i, row in final_df.iterrows():
        if i == 0 or row["Modality_Count"] != final_df.iloc[i - 1]["Modality_Count"]:
            # Find end of this modality count range
            end = i
            while end < len(final_df) - 1 and final_df.iloc[end + 1]["Modality_Count"] == row["Modality_Count"]:
                end += 1
            print(f"  INC_{i + 1:03d} to INC_{end + 1:03d}: "
                  f"{row['Modality_Count']} modalities ({row['Sources_Available']})")

    return final_df


if __name__ == "__main__":
    run_integration()
