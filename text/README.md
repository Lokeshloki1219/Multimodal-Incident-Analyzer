# Text Pipeline - Student 5 (Swet)
## Multimodal Crime / Incident Report Analyzer

### What's Included
- 	ext_pipeline.py - Main pipeline script
- 
equirements.txt - Python dependencies
- 	text_output.csv - Pre-generated output (68 records)
- data/ - CrimeReport dataset (JSONL format)

### Setup Steps (Run on Your System)

1. **Install Python 3.10+**

2. **Create a virtual environment:**
   `ash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   `

3. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   `

4. **Run the pipeline:**
   `ash
   python text_pipeline.py
   `

5. **Output:** 	ext_output.csv with columns:
   Text_ID, Source, Raw_Text, Sentiment, Entities, Topic

### AI Tools Used
- **NLTK** - Text preprocessing, tokenization, stopword removal
- **spaCy** - Named Entity Recognition (people, locations, organizations)
- **HuggingFace Transformers** - Sentiment analysis
- **Zero-Shot Classification** - Topic classification (Theft, Assault, Fire, etc.)

### Dataset
- **CrimeReport** - Real crime text reports with crime type, location, details
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/cameliasiadat/crimereport
- **Already included** in data/ folder

### To Upload to GitHub
1. Place entire 	ext/ folder in repo root
2. Commit and push
