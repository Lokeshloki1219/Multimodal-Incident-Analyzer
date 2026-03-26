"""
Audio Pipeline — Student 1 (Lokeshwar Reddy Peddarangareddy)

Multimodal Crime / Incident Report Analyzer

What it does:
    Converts 911 emergency audio calls into structured data by:
    1. Transcribing audio files to text using OpenAI Whisper
    2. Extracting entities (location, event type) using spaCy NER
    3. Performing sentiment analysis using HuggingFace transformers
    4. Computing an urgency score from sentiment + keyword weights

Input:
    Audio files (.wav, .mp3, .flac) in the 'data/' subdirectory,
    OR uses built-in sample transcripts if no audio files are found.

Output:
    audio_output.csv with columns:
    Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score
"""

import os
import sys
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "audio_output.csv")

# Maximum number of audio files to process (for prototype speed)
# Set to None to process all files
MAX_AUDIO_FILES = 20

# Urgency keyword weights — used to compute Urgency_Score
URGENCY_KEYWORDS = {
    "fire": 0.3, "burning": 0.3, "flames": 0.3,
    "trapped": 0.3, "stuck": 0.2,
    "gun": 0.3, "shot": 0.3, "shooting": 0.3, "weapon": 0.25,
    "help": 0.2, "emergency": 0.2, "hurry": 0.2, "please": 0.1,
    "accident": 0.25, "crash": 0.25, "collision": 0.25,
    "robbery": 0.3, "robbed": 0.3, "stole": 0.25, "theft": 0.25,
    "fight": 0.25, "fighting": 0.25, "assault": 0.3, "attack": 0.3,
    "blood": 0.3, "bleeding": 0.3, "injured": 0.3, "hurt": 0.25,
    "dead": 0.35, "dying": 0.35, "unconscious": 0.3,
    "knife": 0.3, "stabbed": 0.35, "stabbing": 0.35,
    "explosion": 0.35, "bomb": 0.35,
    "scream": 0.2, "screaming": 0.2, "crying": 0.15,
    "breaking": 0.2, "broken": 0.15, "smash": 0.2,
    "smoke": 0.25, "alarm": 0.2,
}

# Confidence threshold for sentiment
CONFIDENCE_THRESHOLD = 0.7

# Sample transcripts — used when no audio files are available
# These simulate realistic 911 emergency calls for different incident types
SAMPLE_TRANSCRIPTS = [
    {
        "transcript": "There is a fire in the building on Downtown Avenue. People are trapped on the second floor. Please send help immediately, there is heavy smoke everywhere.",
        "source": "911_call_001.wav"
    },
    {
        "transcript": "I just witnessed a robbery at the convenience store on Oak Street. Two men with masks ran out with cash. They headed south on foot. One had a gun.",
        "source": "911_call_002.wav"
    },
    {
        "transcript": "There's been a car accident at the intersection of Main Street and Fifth Avenue. Two vehicles collided. One person appears to be injured and is bleeding.",
        "source": "911_call_003.wav"
    },
    {
        "transcript": "I want to report a fight happening outside the bar on Elm Street. There are about five people involved. Someone might have a knife. It's getting really violent.",
        "source": "911_call_004.wav"
    },
    {
        "transcript": "There's a suspicious person breaking into a house on Cedar Lane in the Riverside neighborhood. They smashed the window and climbed in. The homeowners are away on vacation.",
        "source": "911_call_005.wav"
    },
    {
        "transcript": "We need an ambulance at Lincoln Park immediately. A man just collapsed on the ground near the fountain. He's unconscious and not responding. I think he might be having a heart attack.",
        "source": "911_call_006.wav"
    },
    {
        "transcript": "There's a large disturbance at the public square downtown. A crowd of about thirty people is getting aggressive. Some are throwing bottles and there's a lot of screaming.",
        "source": "911_call_007.wav"
    },
    {
        "transcript": "I can hear gunshots coming from the parking garage on Market Street. Multiple shots fired. People are running and screaming. Please send police right away.",
        "source": "911_call_008.wav"
    },
    {
        "transcript": "There's heavy smoke coming from a warehouse on Industrial Boulevard near the river. I can see flames from the roof. No one appears to be inside but the fire is spreading fast.",
        "source": "911_call_009.wav"
    },
    {
        "transcript": "I was just mugged at the bus stop on Washington Avenue near the library. A man grabbed my purse and pushed me to the ground. He ran towards the train station. I'm a bit hurt but okay.",
        "source": "911_call_010.wav"
    },
    {
        "transcript": "A car just hit a pedestrian on Broad Street near the school crossing. The person is lying on the road and not moving. The driver stopped but the pedestrian needs medical attention urgently.",
        "source": "911_call_011.wav"
    },
    {
        "transcript": "I'm calling about a noise complaint on Maple Drive. My neighbors are having an extremely loud party and there seems to be some kind of argument happening. I heard glass breaking.",
        "source": "911_call_012.wav"
    },
]


# ============================================================================
# STEP 1: AUDIO TRANSCRIPTION (Whisper)
# ============================================================================

def load_whisper_model(model_size="base"):
    """Load the OpenAI Whisper model for speech-to-text transcription."""
    try:
        import whisper
        print(f"[Audio] Loading Whisper '{model_size}' model...")
        model = whisper.load_model(model_size)
        print("[Audio] Whisper model loaded successfully.")
        return model
    except ImportError:
        print("[Audio] WARNING: openai-whisper not installed. Using sample transcripts.")
        return None
    except Exception as e:
        print(f"[Audio] WARNING: Could not load Whisper model: {e}")
        print("[Audio] Falling back to sample transcripts.")
        return None


def transcribe_audio_files(model, audio_dir):
    """
    Transcribe audio files in the given directory (recursively) using Whisper.
    Returns a list of dicts with 'transcript' and 'source' keys.
    Limits to MAX_AUDIO_FILES for prototype speed.
    """
    transcripts = []
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = []

    # Recursively search for audio files in subdirectories
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        print(f"[Audio] No audio files found in '{audio_dir}'.")
        return []

    # Limit number of files for prototype
    total_found = len(audio_files)
    if MAX_AUDIO_FILES and len(audio_files) > MAX_AUDIO_FILES:
        audio_files = sorted(audio_files)[:MAX_AUDIO_FILES]
        print(f"[Audio] Found {total_found} audio files. Processing first {MAX_AUDIO_FILES} for prototype.")
    else:
        audio_files = sorted(audio_files)
        print(f"[Audio] Found {total_found} audio file(s). Transcribing all...")

    for i, filepath in enumerate(audio_files):
        filename = os.path.basename(filepath)
        print(f"  [{i+1}/{len(audio_files)}] Transcribing: {filename}...")
        try:
            result = model.transcribe(filepath)
            text = result["text"].strip()
            if text:  # Only include non-empty transcripts
                transcripts.append({
                    "transcript": text,
                    "source": filename
                })
                print(f"    ✓ '{text[:80]}...'")
            else:
                print(f"    ⚠ Empty transcript, skipping.")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    return transcripts


def load_metadata():
    """
    Load the 911 metadata CSV if available.
    Returns a dict mapping filename to metadata row, or empty dict.
    """
    # Look for metadata CSV in data directory and subdirectories
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv') and 'metadata' in f.lower():
                csv_path = os.path.join(root, f)
                print(f"[Audio] Found metadata CSV: {csv_path}")
                try:
                    df = pd.read_csv(csv_path)
                    metadata = {}
                    if 'filename' in df.columns:
                        for _, row in df.iterrows():
                            fn = os.path.basename(str(row.get('filename', '')))
                            metadata[fn] = row.to_dict()
                    print(f"[Audio] Loaded metadata for {len(metadata)} files.")
                    return metadata
                except Exception as e:
                    print(f"[Audio] Could not load metadata: {e}")
    return {}


def get_transcripts():
    """
    Get transcripts either from real audio files (using Whisper) or
    fall back to built-in sample transcripts for demonstration.
    Also loads metadata CSV to enrich transcripts with context.
    """
    # Try to load Whisper and process real audio files
    if os.path.exists(DATA_DIR):
        model = load_whisper_model("base")
        if model is not None:
            transcripts = transcribe_audio_files(model, DATA_DIR)
            if transcripts:
                # Try to enrich with metadata
                metadata = load_metadata()
                if metadata:
                    for t in transcripts:
                        fn = t["source"]
                        if fn in metadata:
                            meta = metadata[fn]
                            # Append metadata description to transcript for richer NER
                            desc = meta.get("description", "")
                            title = meta.get("title", "")
                            state = meta.get("state", "")
                            if desc and isinstance(desc, str):
                                t["transcript"] = t["transcript"] + " " + desc
                            t["metadata_title"] = title if isinstance(title, str) else ""
                            t["metadata_state"] = state if isinstance(state, str) else ""

                print(f"[Audio] Successfully transcribed {len(transcripts)} audio files.")
                return transcripts

    # Fall back to sample transcripts
    print("[Audio] Using built-in sample transcripts for demonstration.")
    print(f"[Audio] {len(SAMPLE_TRANSCRIPTS)} sample transcripts loaded.")
    return SAMPLE_TRANSCRIPTS


# ============================================================================
# STEP 2: ENTITY EXTRACTION (spaCy NER)
# ============================================================================

def load_spacy_model():
    """Load the spaCy English NER model."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("[Audio] spaCy 'en_core_web_sm' model loaded.")
        return nlp
    except OSError:
        print("[Audio] spaCy model not found. Downloading 'en_core_web_sm'...")
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp


def extract_entities(nlp, text):
    """
    Extract location (GPE/LOC/FAC entities) and event type from transcript text.
    Returns (extracted_event, location) tuple.
    """
    doc = nlp(text)

    # Extract location entities
    location_labels = {"GPE", "LOC", "FAC"}
    locations = []
    for ent in doc.ents:
        if ent.label_ in location_labels:
            locations.append(ent.text)

    location = ", ".join(locations) if locations else "Unknown"

    # Extract event type using keyword matching
    text_lower = text.lower()
    event_keywords = {
        "Building fire / trapped persons": ["fire", "burning", "flames", "trapped", "smoke"],
        "Robbery / theft": ["robbery", "robbed", "stole", "theft", "mugged", "purse"],
        "Road accident / collision": ["accident", "crash", "collision", "hit", "vehicle"],
        "Assault / fighting": ["fight", "fighting", "assault", "attack", "violent", "knife", "stabbed"],
        "Shooting / gunfire": ["gun", "shot", "shooting", "gunshot", "fired"],
        "Breaking and entering": ["breaking", "broke in", "burglar", "window", "climbed in"],
        "Medical emergency": ["ambulance", "collapsed", "unconscious", "heart attack", "not responding"],
        "Public disturbance": ["disturbance", "crowd", "aggressive", "throwing", "noise", "loud party"],
        "Explosion": ["explosion", "bomb", "blast", "explode"],
    }

    best_event = "Unknown incident"
    best_score = 0

    for event_name, keywords in event_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_event = event_name

    return best_event, location


# ============================================================================
# STEP 3: SENTIMENT ANALYSIS (HuggingFace Transformers)
# ============================================================================

def load_sentiment_model():
    """Load the HuggingFace sentiment analysis pipeline."""
    try:
        from transformers import pipeline
        print("[Audio] Loading HuggingFace sentiment analysis model...")
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU; change to 0 for GPU
        )
        print("[Audio] Sentiment model loaded.")
        return classifier
    except Exception as e:
        print(f"[Audio] WARNING: Could not load sentiment model: {e}")
        return None


def analyze_sentiment(classifier, text):
    """
    Analyze sentiment of transcript text.
    Returns (sentiment_label, confidence_score) tuple.
    Maps POSITIVE → Calm, NEGATIVE → Distressed.
    """
    if classifier is None:
        # Fallback: keyword-based sentiment
        distress_words = ["help", "emergency", "hurry", "fire", "gun", "shot",
                          "trapped", "blood", "dying", "scream", "please"]
        text_lower = text.lower()
        distress_count = sum(1 for w in distress_words if w in text_lower)
        if distress_count >= 2:
            return "Distressed", 0.85
        elif distress_count == 1:
            return "Distressed", 0.65
        else:
            return "Calm", 0.70

    try:
        # Truncate text to avoid model token limits
        result = classifier(text[:512])[0]
        label = result["label"]
        confidence = result["score"]

        # Map sentiment labels
        if label == "NEGATIVE":
            sentiment = "Distressed"
        else:
            sentiment = "Calm"

        return sentiment, round(confidence, 4)
    except Exception as e:
        print(f"[Audio] Sentiment analysis error: {e}")
        return "Unknown", 0.5


# ============================================================================
# STEP 4: URGENCY SCORE COMPUTATION
# ============================================================================

def compute_urgency_score(text, sentiment_label, sentiment_confidence):
    """
    Compute an urgency score (0.0–1.0) based on:
    - Sentiment confidence (distressed = higher urgency)
    - Presence of urgency keywords in the transcript
    
    Formula: base_score * keyword_boost
    - If distressed: base = sentiment_confidence * 0.5
    - If calm: base = (1 - sentiment_confidence) * 0.3
    - keyword_boost = sum of matched keyword weights (capped at 0.5)
    - Final score = min(base + keyword_boost, 1.0)
    """
    text_lower = text.lower()

    # Base score from sentiment
    if sentiment_label == "Distressed":
        base_score = sentiment_confidence * 0.5
    else:
        base_score = (1 - sentiment_confidence) * 0.3

    # Keyword boost
    keyword_boost = 0.0
    for keyword, weight in URGENCY_KEYWORDS.items():
        if keyword in text_lower:
            keyword_boost += weight

    # Cap keyword boost
    keyword_boost = min(keyword_boost, 0.5)

    # Final urgency score
    urgency = min(base_score + keyword_boost, 1.0)

    return round(urgency, 2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_audio_pipeline():
    """
    Execute the full audio analysis pipeline:
    1. Get transcripts (Whisper or sample data)
    2. Extract entities with spaCy NER
    3. Analyze sentiment with HuggingFace
    4. Compute urgency scores
    5. Export to CSV
    """
    print("=" * 70)
    print("AUDIO PIPELINE — Student 1 (Lokeshwar Reddy)")
    print("Multimodal Crime / Incident Report Analyzer")
    print("=" * 70)

    # Step 1: Get transcripts
    print("\n[Step 1] Getting transcripts...")
    transcripts = get_transcripts()

    if not transcripts:
        print("[Audio] ERROR: No transcripts available. Exiting.")
        sys.exit(1)

    # Step 2: Load NER model
    print("\n[Step 2] Loading spaCy NER model...")
    nlp = load_spacy_model()

    # Step 3: Load sentiment model
    print("\n[Step 3] Loading sentiment analysis model...")
    sentiment_classifier = load_sentiment_model()

    # Step 4: Process each transcript
    print(f"\n[Step 4] Processing {len(transcripts)} transcripts...")
    results = []

    for idx, item in enumerate(transcripts):
        call_id = f"C{(idx + 1):03d}"
        transcript = item["transcript"]

        # Extract entities
        extracted_event, location = extract_entities(nlp, transcript)

        # Analyze sentiment
        sentiment, confidence = analyze_sentiment(sentiment_classifier, transcript)

        # Compute urgency score
        urgency_score = compute_urgency_score(transcript, sentiment, confidence)

        results.append({
            "Call_ID": call_id,
            "Transcript": transcript,
            "Extracted_Event": extracted_event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency_score,
        })

        print(f"  [{call_id}] Event: {extracted_event} | Location: {location} | "
              f"Sentiment: {sentiment} ({confidence}) | Urgency: {urgency_score}")

    # Step 5: Create DataFrame and export
    print(f"\n[Step 5] Exporting results to '{OUTPUT_FILE}'...")
    df = pd.DataFrame(results)

    # Ensure correct column order as per assignment
    df = df[["Call_ID", "Transcript", "Extracted_Event", "Location",
             "Sentiment", "Urgency_Score"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"[Audio] Pipeline complete! Output saved to: {OUTPUT_FILE}")
    print(f"[Audio] Total records: {len(df)}")
    print(f"{'=' * 70}")

    # Display summary
    print("\n--- Output Preview ---")
    print(df.to_string(index=False, max_colwidth=50))

    return df


if __name__ == "__main__":
    run_audio_pipeline()
