"""
LLM Summarizer — AI-powered incident summarization using Flan-T5

Uses HuggingFace's google/flan-t5-base model (free, open-source, runs locally)
to generate 2-3 sentence human-readable summaries for each incident.

Input:  integration/final_incidents.csv
Output: integration/final_incidents.csv (with AI_Summary column appended)

No API key or cloud dependency — model downloads automatically on first run (~990MB).
Processes ~1-2 rows/second on CPU, ~5 minutes for 198 incidents.
"""

import os
import sys
import time
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_CSV = os.path.join(SCRIPT_DIR, "final_incidents.csv")


def load_model():
    """Load google/flan-t5-base for text generation."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    print("[LLM] Loading google/flan-t5-base model...")
    print("[LLM] (First run downloads ~990MB, subsequent runs use cache)")

    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[LLM] Model loaded on {device.upper()}")

    return model, tokenizer, device


def build_prompt(row):
    """Construct a structured prompt from incident fields."""
    parts = []
    parts.append(f"Incident ID: {row['Incident_ID']}")
    parts.append(f"Severity: {row['Severity']}")
    parts.append(f"Sources: {row['Sources_Available']}")
    parts.append(f"Audio Event: {row['Audio_Event']}")
    parts.append(f"PDF Document Type: {row['PDF_Doc_Type']}")
    parts.append(f"Image Objects Detected: {row['Image_Objects']}")
    parts.append(f"Video Event: {row['Video_Event']}")
    parts.append(f"Text Crime Type: {row['Text_Crime_Type']}")

    structured = "\n".join(parts)

    prompt = (
        f"Summarize this multimodal crime incident report in 2-3 sentences. "
        f"Mention the key findings from each available data source.\n\n"
        f"{structured}\n\n"
        f"Summary:"
    )
    return prompt


def generate_summary(model, tokenizer, device, prompt, max_length=150):
    """Generate a summary using Flan-T5."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512,
                       truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


def run_summarizer():
    """Generate AI summaries for all incidents in final_incidents.csv."""
    print("=" * 70)
    print("LLM SUMMARIZER — Flan-T5 Incident Summarization")
    print("Model: google/flan-t5-base (local, no API key needed)")
    print("=" * 70)

    # Load data
    if not os.path.exists(FINAL_CSV):
        print(f"[LLM] ERROR: {FINAL_CSV} not found. Run integrate.py first.")
        sys.exit(1)

    df = pd.read_csv(FINAL_CSV)
    print(f"\n[LLM] Loaded {len(df)} incidents from {FINAL_CSV}")

    # Load model
    model, tokenizer, device = load_model()

    # Generate summaries
    print(f"\n[LLM] Generating AI summaries for {len(df)} incidents...")
    summaries = []
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        prompt = build_prompt(row)
        summary = generate_summary(model, tokenizer, device, prompt)
        summaries.append(summary)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(df) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(df)}] {rate:.1f} rows/sec, "
                  f"~{remaining:.0f}s remaining")
            print(f"    {row['Incident_ID']}: {summary[:80]}...")

    # Add to DataFrame
    df["AI_Summary"] = summaries

    # Save back
    df.to_csv(FINAL_CSV, index=False, encoding="utf-8")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"[LLM] Summarization complete!")
    print(f"[LLM] {len(df)} summaries generated in {total_time:.0f} seconds "
          f"({len(df)/total_time:.1f} rows/sec)")
    print(f"[LLM] AI_Summary column added to {FINAL_CSV}")
    print(f"{'=' * 70}")

    # Preview
    print("\n--- Summary Preview ---")
    for i in [0, 49, 99, 149, 197]:
        if i < len(df):
            row = df.iloc[i]
            print(f"\n  {row['Incident_ID']} (Severity: {row['Severity']}, "
                  f"Sources: {row['Modality_Count']}):")
            print(f"    {row['AI_Summary']}")

    return df


if __name__ == "__main__":
    run_summarizer()
