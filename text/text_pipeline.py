"""
Text/NLP Pipeline — Student 5 (Swet Vimalkumar Patel)

Multimodal Crime / Incident Report Analyzer

What it does:
    Analyzes crime-related news articles and text reports by:
    1. Preprocessing text using NLTK (tokenization, stopword removal)
    2. Extracting named entities (persons, locations, orgs) using spaCy NER
    3. Performing zero-shot topic classification using HuggingFace
    4. Analyzing sentiment using HuggingFace sentiment pipeline
    5. Generating structured output

Input:
    CSV files with news articles in the 'data/' subdirectory,
    OR uses built-in sample news articles if no data files are found.

Output:
    text_output.csv with columns:
    Text_ID, Source, Raw_Text, Sentiment, Entities, Topic
"""

import os
import sys
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "text_output.csv")

# Maximum number of articles to process (None = all)
MAX_ARTICLES = None

# Zero-shot classification labels
# Aligned with assignment requirement: "topic classification (accident / fire / theft / disturbance)"
# Removed "Police Operation" — too generic; most crime tweets mention police,
# causing 70% of records to land in this catch-all category.
TOPIC_LABELS = [
    "Murder / Homicide",
    "Robbery / Theft",
    "Assault / Violence",
    "Drug Crime",
    "Traffic Accident",
    "Fraud / White-collar Crime",
    "Kidnapping / Missing Person",
    "Arson / Fire",
    "Public Disturbance",
    "Sexual Assault",
    "Gang Violence",
    "Shooting",
    "Law Enforcement / Policing",
]


# Sample news articles for fallback
SAMPLE_ARTICLES = [
    {
        "title": "Two shot dead in drive-by shooting in Laventille",
        "text": "Two men were killed in a drive-by shooting in Laventille early Thursday morning. Police say the victims, identified as Marcus James, 28, and David Smith, 31, were standing at a street corner when gunmen opened fire from a passing vehicle. Officers from the Homicide Bureau are investigating.",
        "source": "crime_news"
    },
    {
        "title": "Armed robbery at downtown bank leaves three injured",
        "text": "Three people were injured during an armed robbery at First National Bank in Port of Spain. Two masked men armed with handguns entered the bank at approximately 2:30 PM and demanded cash. A security guard was shot in the leg during the incident. Police have launched a manhunt for the suspects.",
        "source": "crime_news"
    },
    {
        "title": "Drug bust nets $5 million in cocaine at port facility",
        "text": "Officers from the Organized Crime and Intelligence Unit seized approximately 50 kilograms of cocaine valued at $5 million during a raid at a port facility in San Fernando. Three suspects were arrested and charged with possession with intent to distribute.",
        "source": "crime_news"
    },
    {
        "title": "Hit and run accident kills pedestrian on highway",
        "text": "A 45-year-old woman was killed in a hit-and-run accident on the Solomon Hochoy Highway near Chaguanas. The victim was attempting to cross the highway when she was struck by a vehicle traveling at high speed. The driver fled the scene. Police are reviewing CCTV footage.",
        "source": "crime_news"
    },
    {
        "title": "Teen stabbed during school fight in San Juan",
        "text": "A 16-year-old student was hospitalized after being stabbed during a fight at a secondary school in San Juan. The incident occurred during the lunch period. The suspect, a 17-year-old student, has been detained by police. The victim is in stable condition.",
        "source": "crime_news"
    },
    {
        "title": "Police operation dismantles kidnapping ring",
        "text": "A joint police and military operation led to the arrest of five members of a kidnapping ring operating in the East-West Corridor. The gang was responsible for at least three kidnappings in the past six months. Officers rescued a businessman who had been held captive for four days.",
        "source": "crime_news"
    },
    {
        "title": "Arson suspected in warehouse fire that caused millions in damage",
        "text": "Fire investigators believe a blaze that destroyed a warehouse in Chaguanas was deliberately set. The fire caused an estimated $10 million in damage. Accelerants were found at the scene. No injuries were reported. Police are looking for two suspects.",
        "source": "crime_news"
    },
    {
        "title": "Woman charged with fraud after $2 million Ponzi scheme",
        "text": "A 42-year-old businesswoman from Woodbrook was charged with fraud after operating a Ponzi scheme that defrauded over 200 investors of approximately $2 million. The accused promised returns of 15% per month on investments.",
        "source": "crime_news"
    },
    {
        "title": "Gang shootout in Beetham Gardens leaves four injured",
        "text": "Four people including a 12-year-old boy were injured in a gang-related shootout in Beetham Gardens. Rival gangs exchanged gunfire for approximately 20 minutes before police arrived. The injured were taken to Port of Spain General Hospital.",
        "source": "crime_news"
    },
    {
        "title": "Sexual assault reported at nightclub in St. James",
        "text": "Police are investigating a sexual assault that allegedly occurred at a nightclub on Ariapita Avenue in St. James. The 23-year-old victim reported the incident to police after seeking medical attention. A 30-year-old man has been taken into custody.",
        "source": "crime_news"
    },
    {
        "title": "Domestic dispute leads to fatal stabbing in Tunapuna",
        "text": "A man was fatally stabbed during a domestic dispute at a residence in Tunapuna. The victim, identified as Roger Williams, 35, was pronounced dead at the scene. His common-law wife, 32, was detained by officers and is assisting with investigations.",
        "source": "crime_news"
    },
    {
        "title": "Police seize illegal firearms cache in Morvant raid",
        "text": "Officers from the Special Operations Response Team discovered a cache of illegal firearms during a predawn raid in Morvant. The haul included two AR-15 rifles, three handguns, and over 500 rounds of ammunition. Two men were arrested at the scene.",
        "source": "crime_news"
    },
]


# ============================================================================
# STEP 1: TEXT PREPROCESSING (NLTK)
# ============================================================================

def setup_nltk():
    """Download required NLTK data."""
    try:
        import nltk
        for resource in ['punkt', 'stopwords', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
        print("[Text] NLTK resources loaded.")
    except ImportError:
        print("[Text] WARNING: NLTK not installed.")


def preprocess_text(text):
    """Clean and preprocess text using NLTK."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        # Basic cleaning
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        # Tokenize and remove stopwords for analysis
        try:
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
            return text, filtered
        except Exception:
            return text, text.lower().split()

    except ImportError:
        # Fallback without NLTK
        text = re.sub(r'\s+', ' ', text).strip()
        return text, text.lower().split()


# ============================================================================
# STEP 2: NAMED ENTITY RECOGNITION (spaCy)
# ============================================================================

def load_spacy_model():
    """Load the spaCy English NER model."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("[Text] spaCy 'en_core_web_sm' model loaded.")
        return nlp
    except OSError:
        print("[Text] Downloading spaCy model...")
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp


def extract_entities(nlp, text):
    """
    Extract key entities from text using spaCy NER.
    Returns (formatted_string, location_list) tuple.
    """
    doc = nlp(text[:2000])  # Limit text length

    entities = {
        "PERSON": set(),
        "GPE": set(),     # Countries, cities, states
        "ORG": set(),     # Organizations
        "LOC": set(),     # Locations
        "DATE": set(),
    }

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].add(ent.text.strip())

    # Build formatted string
    parts = []
    persons = list(entities["PERSON"])[:3]
    locations = list(entities["GPE"] | entities["LOC"])[:3]
    orgs = list(entities["ORG"])[:2]

    if persons:
        parts.append("Persons: " + ", ".join(persons))
    if locations:
        parts.append("Locations: " + ", ".join(locations))
    if orgs:
        parts.append("Orgs: " + ", ".join(orgs))

    formatted = "; ".join(parts) if parts else "No key entities found"
    return formatted, locations


def extract_location_entity(ner_locations, place_info=None):
    """
    Extract the primary location entity for the Location_Entity column.
    Uses NER locations and optionally the tweet's place field.
    """
    # Prefer tweet place data (from CrimeReport dataset)
    if place_info:
        return place_info

    # Use NER-extracted locations
    if ner_locations:
        return ", ".join(ner_locations[:2])

    return "Unknown"


def classify_severity(text, topic):
    """
    Classify incident severity as High / Medium / Low.
    Based on keyword presence and topic type.
    """
    text_lower = text.lower()

    high_keywords = [
        "killed", "dead", "murder", "homicide", "shot", "shooting",
        "stabbed", "fatal", "critical", "died", "death", "armed",
        "hostage", "kidnap", "carjack", "explosion"
    ]
    medium_keywords = [
        "injured", "arrested", "robbery", "assault", "theft",
        "burglary", "fight", "drug", "seized", "fire", "arson",
        "crash", "accident", "suspect"
    ]

    high_topics = ["Murder / Homicide", "Shooting", "Gang Violence",
                   "Kidnapping / Missing Person", "Sexual Assault"]
    medium_topics = ["Robbery / Theft", "Assault / Violence", "Arson / Fire",
                     "Drug Crime"]

    high_score = sum(1 for kw in high_keywords if kw in text_lower)
    med_score = sum(1 for kw in medium_keywords if kw in text_lower)

    if high_score >= 2 or topic in high_topics:
        return "High"
    elif med_score >= 2 or topic in medium_topics:
        return "Medium"
    else:
        return "Low"


# ============================================================================
# STEP 3: ZERO-SHOT TOPIC CLASSIFICATION (HuggingFace)
# ============================================================================

def load_classifier():
    """Load HuggingFace zero-shot classification pipeline."""
    try:
        from transformers import pipeline
        print("[Text] Loading zero-shot classification model (bart-large-mnli)...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        print("[Text] Zero-shot classifier loaded.")
        return classifier
    except Exception as e:
        print(f"[Text] WARNING: Could not load zero-shot classifier: {e}")
        print("[Text] Falling back to keyword-based classification.")
        return None


def classify_topic(classifier, text, title=""):
    """
    Classify the topic of a text using zero-shot classification.
    Falls back to keyword-based classification if model unavailable.
    """
    combined = f"{title}. {text[:500]}" if title else text[:500]

    if classifier is not None:
        try:
            result = classifier(combined, TOPIC_LABELS, multi_label=False)
            return result["labels"][0]
        except Exception as e:
            print(f"    [ZeroShot] Error: {e}")

    # Keyword-based fallback
    text_lower = combined.lower()
    keyword_map = {
        "Murder / Homicide": ["killed", "murder", "homicide", "dead", "fatally", "death"],
        "Robbery / Theft": ["robbery", "robbed", "theft", "stole", "armed robbery", "heist"],
        "Assault / Violence": ["assault", "stabbed", "stabbing", "fight", "beaten", "attack"],
        "Drug Crime": ["drug", "cocaine", "marijuana", "narcotics", "trafficking"],
        "Traffic Accident": ["accident", "crash", "hit-and-run", "collision", "highway"],
        "Fraud / White-collar Crime": ["fraud", "ponzi", "scam", "embezzlement"],
        "Kidnapping / Missing Person": ["kidnap", "missing", "abducted", "ransom"],
        "Arson / Fire": ["arson", "fire", "blaze", "burned", "accelerant"],
        "Gang Violence": ["gang", "shootout", "territory", "rival"],
        "Sexual Assault": ["sexual assault", "rape", "molest"],
        "Police Operation": ["police operation", "raid", "seized", "bust", "arrested"],
        "Public Disturbance": ["disturbance", "riot", "protest", "unrest"],
    }

    best_topic = "General Crime"
    best_score = 0

    for topic, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


# ============================================================================
# STEP 4: SENTIMENT ANALYSIS
# ============================================================================

def load_sentiment_model():
    """Load HuggingFace sentiment analysis pipeline."""
    try:
        from transformers import pipeline
        print("[Text] Loading sentiment analysis model...")
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        print("[Text] Sentiment model loaded.")
        return classifier
    except Exception as e:
        print(f"[Text] WARNING: Could not load sentiment model: {e}")
        return None


def analyze_sentiment(classifier, text):
    """Analyze sentiment: returns 'Negative', 'Positive', or 'Neutral'."""
    if classifier is not None:
        try:
            result = classifier(text[:512])[0]
            label = result["label"]
            if label == "NEGATIVE":
                return "Negative"
            else:
                return "Positive"
        except Exception:
            pass

    # Fallback
    negative_words = ["killed", "dead", "shot", "stabbed", "injured", "arrested",
                      "crime", "theft", "robbery", "assault", "murder", "fire"]
    text_lower = text.lower()
    neg_count = sum(1 for w in negative_words if w in text_lower)
    return "Negative" if neg_count >= 2 else "Neutral"


# ============================================================================
# STEP 5: LOAD TEXT DATA
# ============================================================================

def load_jsonl_file(file_path):
    """Load articles from a JSONL (JSON Lines) file like CrimeReport."""
    import json
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    if text and len(text) > 50:
                        # Extract place info from CrimeReport tweet format
                        place_info = None
                        place_obj = obj.get('place')
                        if place_obj and isinstance(place_obj, dict):
                            place_info = place_obj.get('full_name', '')

                        articles.append({
                            "title": "",
                            "text": text,
                            "source": "CrimeReport",
                            "place": place_info,
                            "created_at": obj.get('created_at', ''),
                        })
                except json.JSONDecodeError:
                    continue
        print(f"[Text] Loaded {len(articles)} articles from JSONL: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"[Text] Error loading JSONL: {e}")
    return articles


def load_text_data():
    """
    Load text articles from data directory (CSV, JSONL, or TXT files).
    Falls back to sample data if no data found.
    Returns list of dicts with 'title', 'text', 'source' keys.
    """
    articles = []

    # Try loading files from data directory
    if os.path.exists(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                file_path = os.path.join(root, f)

                # Load JSONL/TXT files (CrimeReport format)
                if f.endswith('.txt') or f.endswith('.jsonl'):
                    jsonl_articles = load_jsonl_file(file_path)
                    articles.extend(jsonl_articles)

                # Load CSV files
                elif f.endswith('.csv'):
                    print(f"[Text] Loading CSV: {file_path}")
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        text_col = None
                        title_col = None

                        for col in ['article', 'text', 'content', 'body', 'description']:
                            if col in df.columns:
                                text_col = col
                                break

                        for col in ['title', 'headline', 'heading']:
                            if col in df.columns:
                                title_col = col
                                break

                        if text_col:
                            for _, row in df.iterrows():
                                text = str(row.get(text_col, ''))
                                if len(text) > 50:
                                    articles.append({
                                        "title": str(row.get(title_col, '')) if title_col else "",
                                        "text": text,
                                        "source": "CrimeReport"
                                    })
                            print(f"[Text] Loaded {len(articles)} articles from {f}")

                    except Exception as e:
                        print(f"[Text] Error loading {f}: {e}")

    if articles:
        # Limit and return
        if MAX_ARTICLES and len(articles) > MAX_ARTICLES:
            articles = articles[:MAX_ARTICLES]
            print(f"[Text] Limited to {MAX_ARTICLES} articles for prototype.")
        return articles

    # Fall back to sample data
    print("[Text] No data files found. Using built-in sample articles.")
    print(f"[Text] {len(SAMPLE_ARTICLES)} sample articles loaded.")
    return SAMPLE_ARTICLES


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_text_pipeline():
    """
    Execute the full text/NLP analysis pipeline:
    1. Load and preprocess text articles
    2. Extract entities with spaCy NER
    3. Classify topics with zero-shot / keyword matching
    4. Analyze sentiment
    5. Export to CSV
    """
    print("=" * 70)
    print("TEXT/NLP PIPELINE — Student 5 (Swet Vimalkumar Patel)")
    print("Multimodal Crime / Incident Report Analyzer")
    print("=" * 70)

    # Step 1: Setup and load data
    print("\n[Step 1] Loading text data...")
    setup_nltk()
    articles = load_text_data()

    if not articles:
        print("[Text] ERROR: No articles available. Exiting.")
        sys.exit(1)

    # Step 2: Load NLP models
    print("\n[Step 2] Loading NLP models...")
    nlp = load_spacy_model()
    topic_classifier = load_classifier()
    sentiment_classifier = load_sentiment_model()

    # Step 3: Process each article
    print(f"\n[Step 3] Processing {len(articles)} articles...")
    results = []

    for idx, article in enumerate(articles):
        text_id = f"TXT_{(idx + 1):03d}"
        title = article.get("title", "")
        text = article["text"]
        source = article.get("source", "unknown")

        print(f"\n  [{text_id}] {title[:60]}..." if len(title) > 60
              else f"\n  [{text_id}] {title}")

        # Preprocess
        clean_text, tokens = preprocess_text(text)

        # Extract entities (now returns tuple)
        entities_str, ner_locations = extract_entities(nlp, clean_text)

        # Extract location entity (from NER + tweet place data)
        place_info = article.get("place", None)
        location_entity = extract_location_entity(ner_locations, place_info)

        # Classify topic
        topic = classify_topic(topic_classifier, clean_text, title)

        # Derive Crime_Type from Topic (assignment Table 7 requires this)
        crime_type = topic  # Topic is already the crime classification

        # Analyze sentiment
        sentiment = analyze_sentiment(sentiment_classifier, clean_text)

        # Classify severity
        severity_label = classify_severity(clean_text, topic)

        results.append({
            "Text_ID": text_id,
            "Source": source,
            "Raw_Text": text[:300],  # Original unprocessed text (truncated for CSV)
            "Crime_Type": crime_type,
            "Location_Entity": location_entity,
            "Sentiment": sentiment,
            "Entities": entities_str,
            "Topic": topic,
            "Severity_Label": severity_label,
        })

        print(f"    Topic: {topic} | Crime: {crime_type} | Severity: {severity_label}")
        print(f"    Location: {location_entity} | Sentiment: {sentiment}")

    # Step 4: Export to CSV
    print(f"\n[Step 4] Exporting results to '{OUTPUT_FILE}'...")
    df = pd.DataFrame(results)

    # Column order per assignment spec
    df = df[["Text_ID", "Source", "Raw_Text", "Crime_Type", "Location_Entity",
             "Sentiment", "Entities", "Topic", "Severity_Label"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"[Text] Pipeline complete! Output saved to: {OUTPUT_FILE}")
    print(f"[Text] Total records: {len(df)}")
    print(f"{'=' * 70}")

    # Display summary
    print("\n--- Output Preview ---")
    print(df.to_string(index=False, max_colwidth=50))

    return df


if __name__ == "__main__":
    run_text_pipeline()
