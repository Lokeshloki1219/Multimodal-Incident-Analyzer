"""
Download the 911 audio dataset from Kaggle using the API token.
Reads KAGGLE_API_TOKEN from environment variable — no hardcoded keys.
"""

import os
import sys
import json
import zipfile
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# The Kaggle notebook/dataset for 911 calls
# We'll search for audio datasets related to 911 calls
DATASET_SLUG = "stpeteishii/911-calls-wav2vec2"
KERNEL_SLUG = "stpeteishii/911-calls-wav2vec2"


def get_token():
    """Read API token from environment variable."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("ERROR: KAGGLE_API_TOKEN environment variable not set.")
        print("Set it with: $env:KAGGLE_API_TOKEN='your_token_here'")
        sys.exit(1)
    return token


def search_datasets(token, query="911 emergency calls audio"):
    """Search Kaggle for relevant audio datasets."""
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://www.kaggle.com/api/v1/datasets/list"
    params = {"search": query, "maxSize": 500 * 1024 * 1024}  # max 500MB

    print(f"[Kaggle] Searching datasets for: '{query}'")
    resp = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        print(f"[Kaggle] Search failed: {resp.status_code} - {resp.text[:200]}")
        return []

    datasets = resp.json()
    print(f"[Kaggle] Found {len(datasets)} datasets:")
    for ds in datasets[:10]:
        ref = ds.get("ref", "unknown")
        title = ds.get("title", "unknown")
        size = ds.get("totalBytes", 0) / (1024 * 1024)
        print(f"  - {ref} | {title} | {size:.1f} MB")

    return datasets


def download_dataset(token, dataset_ref, output_dir):
    """Download a Kaggle dataset by its reference slug."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "dataset.zip")

    print(f"\n[Kaggle] Downloading dataset: {dataset_ref}")
    print(f"[Kaggle] This may take a few minutes...")

    resp = requests.get(url, headers=headers, stream=True)

    if resp.status_code != 200:
        print(f"[Kaggle] Download failed: {resp.status_code} - {resp.text[:200]}")
        return False

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r[Kaggle] Downloading: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

    print(f"\n[Kaggle] Download complete: {zip_path}")

    # Extract
    print(f"[Kaggle] Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)
    print(f"[Kaggle] Extraction complete!")

    # List extracted files
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for fn in filenames:
            filepath = os.path.join(root, fn)
            size = os.path.getsize(filepath) / 1024
            rel = os.path.relpath(filepath, output_dir)
            files.append(rel)
            print(f"  {rel} ({size:.1f} KB)")

    print(f"\n[Kaggle] Total files extracted: {len(files)}")
    return True


def download_kernel_output(token, kernel_ref, output_dir):
    """Download output files from a Kaggle notebook/kernel."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://www.kaggle.com/api/v1/kernels/output/{kernel_ref}"

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "kernel_output.zip")

    print(f"\n[Kaggle] Downloading kernel output: {kernel_ref}")

    resp = requests.get(url, headers=headers, stream=True)

    if resp.status_code != 200:
        print(f"[Kaggle] Kernel output download failed: {resp.status_code}")
        print(f"[Kaggle] Trying dataset download instead...")
        return False

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r[Kaggle] Downloading: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

    print(f"\n[Kaggle] Download complete: {zip_path}")

    # Extract
    print(f"[Kaggle] Extracting to {output_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        os.remove(zip_path)
        print(f"[Kaggle] Extraction complete!")
        return True
    except zipfile.BadZipFile:
        # Might be a single file, not a zip
        new_name = os.path.join(output_dir, "kernel_output.txt")
        os.rename(zip_path, new_name)
        print(f"[Kaggle] Output saved as: {new_name}")
        return True


if __name__ == "__main__":
    token = get_token()

    # First try: search for 911 audio datasets
    print("=" * 60)
    print("STEP 1: Searching for 911 emergency audio datasets")
    print("=" * 60)
    datasets = search_datasets(token, "911 emergency calls audio wav")

    # Also search for general emergency audio
    if len(datasets) < 3:
        datasets.extend(search_datasets(token, "emergency audio calls"))

    # Try downloading the kernel output (the Wav2Vec2 notebook)
    print("\n" + "=" * 60)
    print("STEP 2: Downloading kernel output from 911-calls-wav2vec2")
    print("=" * 60)
    success = download_kernel_output(token, KERNEL_SLUG, DATA_DIR)

    if not success:
        # Fallback: search and download a dataset
        print("\n" + "=" * 60)
        print("STEP 3: Trying direct dataset downloads")
        print("=" * 60)

        # Try some known 911/emergency audio datasets
        fallback_datasets = [
            "osanseviero/emergency-audio",
            "warcoder/emergency-vehicles-sirens-sounds",
        ]

        for ds in fallback_datasets:
            print(f"\nTrying: {ds}")
            if download_dataset(token, ds, DATA_DIR):
                break
        else:
            print("\n[Kaggle] Could not download audio dataset automatically.")
            print("[Kaggle] The sample transcripts in audio_pipeline.py will be used instead.")

    print("\n" + "=" * 60)
    print("DONE! Check the audio/data/ folder for downloaded files.")
    print("=" * 60)
