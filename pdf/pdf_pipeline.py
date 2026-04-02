"""
PDF/Document Pipeline — Student 2 (Neha Reddy Poreddy)

Multimodal Crime / Incident Report Analyzer

What it does:
    Extracts structured information from police report PDFs by:
    1. Extracting raw text from PDF files using PyMuPDF (fitz) as primary
       extractor, with pdfplumber as secondary extractor
    2. Applying OCR fallback (pytesseract) for scanned documents
    3. Running spaCy NER for entity extraction (dates, locations, names)
    4. Classifying incident types using keyword matching
    5. Extracting suspect descriptions and outcomes
    6. Generating structured summaries

Input:
    PDF files in the 'data/' subdirectory,
    OR uses built-in sample police reports if no PDFs are found.

Output:
    pdf_output.csv with columns:
    Report_ID, Incident_Type, Date, Location, Officer, Summary
"""

import os
import sys
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "pdf_output.csv")

# Sample police report texts — used when no PDF files are available
# These simulate realistic police/incident documents for different incident types
SAMPLE_REPORTS = [
    {
        "text": """ARKANSAS POLICE DEPARTMENT
1033 Training Plan Proposal

Date: April 10, 2015
Department: Little Rock Police Department
Location: Little Rock, Arkansas
Requesting Officer: Captain James Morrison

Subject: Request for Tactical Equipment and Training

This proposal outlines the department's request under the 1033 Program for surplus
military equipment to support law enforcement operations. The department is seeking
tactical gear including body armor, night vision goggles, and armored vehicles.

The training plan includes a 40-hour certification course for all officers assigned
to the tactical unit. Training will cover equipment maintenance, proper deployment
procedures, and community engagement protocols.

Equipment Requested:
- 12 sets of Level III body armor
- 4 night vision devices (AN/PVS-14)
- 1 Mine-Resistant Ambush Protected (MRAP) vehicle
- 24 tactical helmets

Budget: $45,000 (training costs only; equipment provided through 1033 Program)
Approval Status: Pending review by State Coordinator
""",
        "filename": "1033_training_proposal_littlerock.pdf"
    },
    {
        "text": """INCIDENT REPORT — FIRE INVESTIGATION

Report Number: FR-2024-00892
Date of Incident: March 15, 2024
Time: 02:45 AM
Location: 1420 Industrial Boulevard, Pine Bluff, Arkansas

Reporting Officer: Lieutenant Sarah Chen, Fire Marshal Division
Investigating Unit: Arson Investigation Squad

Description of Incident:
At approximately 02:45 AM, fire units responded to a structure fire at an abandoned
warehouse at 1420 Industrial Boulevard. Upon arrival, the building was fully engulfed
in flames. Fire crews from Station 7 and Station 12 worked to contain the blaze.

Evidence of accelerant was found at multiple entry points, suggesting intentional
ignition. Security camera footage from the adjacent business shows two unidentified
individuals approaching the building at approximately 02:30 AM.

Suspect Description: Two males, approximately 5'10" to 6'0", wearing dark clothing
and face coverings. Last seen heading east on Industrial Boulevard on foot.

Outcome: Case classified as arson. Investigation ongoing. No arrests made.
Estimated Damage: $350,000
Injuries: None (building was unoccupied)
""",
        "filename": "fire_investigation_report.pdf"
    },
    {
        "text": """POLICE DEPARTMENT — THEFT REPORT

Case Number: TH-2024-02341
Date Filed: February 28, 2024
Date of Incident: February 27, 2024
Time of Incident: 11:30 PM
Location: QuickMart Convenience Store, 890 Oak Street, Fort Smith, Arkansas

Reporting Officer: Officer David Park, Badge #4521
Victim: QuickMart Inc. (Manager: Robert Williams)

Narrative:
On February 27, 2024, at approximately 11:30 PM, two masked individuals entered
the QuickMart Convenience Store at 890 Oak Street. Suspect 1 brandished a handgun
and demanded cash from the register. Suspect 2 proceeded to take merchandise from
the shelves including cigarettes and electronics.

The suspects fled the scene in a dark-colored sedan heading northbound on Oak Street.
Total estimated loss is $2,400 in cash and $800 in merchandise.

Surveillance footage has been secured. Witness statements collected from store clerk
Maria Santos and customer John Doe (name changed for privacy).

Suspect 1: Male, approximately 6'1", medium build, black hoodie, armed with handgun
Suspect 2: Male, approximately 5'8", slim build, gray jacket

Outcome: Under investigation. Evidence submitted to forensics lab.
Follow-up: Detectives assigned. BOLO issued for dark sedan.
""",
        "filename": "theft_report_quickmart.pdf"
    },
    {
        "text": """TRAFFIC ACCIDENT REPORT

Report Number: TA-2024-01567
Date: January 14, 2024
Time: 08:15 AM
Location: Intersection of Main Street and 5th Avenue, Fayetteville, Arkansas

Investigating Officer: Sergeant Michael Torres, Traffic Division

Vehicle 1: 2021 Toyota Camry (Driver: Emily Johnson, age 34)
Vehicle 2: 2019 Ford F-150 (Driver: Thomas Brown, age 52)

Description:
Vehicle 1 was traveling westbound on Main Street. Vehicle 2 was traveling
southbound on 5th Avenue. Vehicle 2 ran a red light and collided with Vehicle 1
at the intersection. Vehicle 1 sustained major driver-side damage. Vehicle 2
sustained front-end damage.

Injuries:
- Emily Johnson: Transported to Washington Regional Medical Center with a
  fractured left arm and minor lacerations. Condition: Stable.
- Thomas Brown: Minor bruising from seatbelt. Refused medical transport.

Citations: Thomas Brown cited for running a red light (ACA 27-52-104)
and failure to yield at intersection.

Weather Conditions: Clear, dry pavement
Road Conditions: Good
Contributing Factors: Red light violation by Vehicle 2

Outcome: Thomas Brown at fault. Insurance claim filed. No criminal charges.
""",
        "filename": "traffic_accident_report.pdf"
    },
    {
        "text": """ASSAULT INCIDENT REPORT

Case Number: AS-2024-00456
Date: March 3, 2024
Time: 10:45 PM
Location: Riverside Bar & Grill, 234 Elm Street, Jonesboro, Arkansas

Reporting Officer: Officer Angela Rivera, Badge #3189
Assisting Officer: Officer Keith Washington, Badge #3204

Victim: Marcus Thompson, age 28, resident of Jonesboro
Suspect: Derek Sullivan, age 31, resident of Paragould, Arkansas

Narrative:
Officers responded to a disturbance call at Riverside Bar & Grill at
approximately 10:45 PM. Upon arrival, officers found the victim, Marcus Thompson,
with visible injuries to the face and head area. Witnesses stated that the suspect,
Derek Sullivan, initiated a physical altercation after a verbal dispute over a
pool game.

Sullivan was observed by multiple witnesses striking Thompson repeatedly with
his fists and then using a beer bottle, causing a laceration above the victim's
left eye. Bartender Jessica Miles called 911 during the incident.

Thompson was transported to St. Bernards Medical Center for treatment of facial
lacerations and a possible concussion.

Suspect Sullivan was located at the scene and placed under arrest for assault
in the second degree (ACA 5-13-206). Sullivan was visibly intoxicated.

Outcome: Suspect arrested. Charged with assault in the second degree
and public intoxication. Bail set at $5,000.
""",
        "filename": "assault_report_jonesboro.pdf"
    },
    {
        "text": """DEPARTMENT OF LAW ENFORCEMENT
TRAINING EQUIPMENT INVENTORY REPORT

Report Number: INV-2024-0078
Date: January 30, 2024
Department: Springdale Police Department
Location: Springdale, Arkansas
Prepared by: Quartermaster Sergeant Lisa Nguyen

Annual Equipment Audit Summary:

This report documents the current inventory status of all training equipment
assigned to the Springdale Police Department under federal programs including
the 1033 Program and Byrne JAG grants.

Current Inventory:
- 8 sets Level IIIA body armor (serviceable)
- 2 sets Level IV body armor (needs replacement)
- 6 patrol rifles (AR-15 platform, fully operational)
- 12 less-lethal launchers (bean bag/pepper ball)
- 3 tactical shields (ballistic rated)
- 1 mobile command vehicle (operational, due for maintenance Q3)
- 45 training ammunition cases (sufficient for 6 months)

Training Hours Completed (FY2024): 2,400 hours across 80 officers
Certification Status: All tactical team members current
Next Audit Date: July 30, 2024

Recommendations:
Replace aging Level IV body armor sets. Request additional night vision
equipment for rural patrol operations. Schedule advanced tactical training
for new recruits joining in Q2 2024.
""",
        "filename": "equipment_inventory_springdale.pdf"
    },
    {
        "text": """PUBLIC DISTURBANCE REPORT

Report Number: PD-2024-00234
Date: February 15, 2024
Time: 09:30 PM
Location: Central Park, Downtown Square, Hot Springs, Arkansas

Reporting Officer: Sergeant William Hayes, Badge #2876
Units Responding: Patrol Units 14, 17, 22

Description:
Multiple 911 calls reported a large gathering of approximately 50 persons
at Central Park that had become disorderly. Upon arrival, officers observed
approximately 40-50 individuals engaged in a loud confrontation. Several
participants were throwing glass bottles and other debris.

Crowd dispersal was initiated using verbal commands and the PA system on
Unit 22. The majority of the crowd dispersed within 15 minutes. Six
individuals who refused to leave were detained.

Arrests:
1. Jason Williams, age 22 — Disorderly conduct, resisting arrest
2. Tyler Mitchell, age 19 — Disorderly conduct, public intoxication
3. Brandon Lopez, age 24 — Disorderly conduct

Three additional individuals issued citations for noise violations.

Property Damage: Two park benches damaged, estimated repair cost $1,200.
Injuries: Officer Hayes sustained minor injury to left hand during
arrest of Jason Williams. Treated on scene.

Outcome: Situation resolved. Three arrests, three citations issued.
""",
        "filename": "public_disturbance_hotsprings.pdf"
    },
    {
        "text": """SHOOTING INCIDENT REPORT

Case Number: SH-2024-00089
Date: March 8, 2024
Time: 01:15 AM
Location: 1800 Block of Market Street, Texarkana, Arkansas

Reporting Officer: Detective Frank Morgan, Homicide Division
Assisting Units: Patrol Units 5, 9; Crime Scene Unit; K-9 Unit

Victim: Deshawn Parker, age 25, Texarkana resident
Suspect: Unknown at time of report

Narrative:
Officers responded to a report of shots fired at 1800 block of Market Street.
Upon arrival, officers located the victim, Deshawn Parker, in the parking lot
of the Texarkana Shopping Plaza with multiple gunshot wounds to the torso
and lower extremities.

EMS transported the victim to Wadley Regional Medical Center where he was
listed in critical condition. Surgery was performed to address internal injuries.

Witnesses reported hearing approximately 6-8 gunshots. One witness (anonymous)
reported seeing a black SUV flee the scene heading eastbound on Market Street
immediately after the shooting. Shell casings (9mm) recovered at the scene.

Evidence Collected:
- 7 shell casings (9mm Luger)
- Surveillance footage from Shopping Plaza cameras
- Witness statements (3 witnesses interviewed)
- Victim's personal effects

Suspect Description: Unknown. Vehicle: Black SUV, possibly Chevrolet Tahoe
or similar, no visible plates.

Outcome: Active investigation. Crime Stoppers tip line activated.
Victim upgraded to stable condition on March 10, 2024.
""",
        "filename": "shooting_report_texarkana.pdf"
    },
    {
        "text": """BURGLARY / BREAKING AND ENTERING REPORT

Case Number: BE-2024-01782
Date: February 5, 2024
Time of Discovery: 07:00 AM (estimated time of entry: 02:00-04:00 AM)
Location: Riverside Community Center, 456 Cedar Lane, Conway, Arkansas

Reporting Officer: Officer Patricia Kim, Badge #5102
Investigating Detective: Detective Ray Sanchez

Victim: Conway Parks & Recreation Department

Narrative:
Staff member Robert Anderson arrived at the Riverside Community Center at
7:00 AM and discovered the rear entrance had been forcibly entered. A window
adjacent to the back door was smashed, and the door was found unlocked from
the inside.

Items Stolen:
- 2 laptop computers (Dell, estimated value $1,800)
- 1 projector (Epson, estimated value $650)
- Cash from donation box (approximately $300)
- 1 set of master keys for the facility

Damage:
- Rear window broken (replacement cost $400)
- Interior office door lock damaged
- Filing cabinet forced open (contents scattered, no sensitive documents missing)

Forensic Evidence: Partial fingerprints recovered from window frame and filing
cabinet. Shoe impressions photographed near rear entrance.

Suspect Description: Unknown. Evidence suggests single perpetrator.

Outcome: Under investigation. Fingerprints submitted to AFIS database.
Security camera footage being reviewed. Locks changed and security upgraded.
""",
        "filename": "burglary_report_conway.pdf"
    },
    {
        "text": """NARCOTICS INVESTIGATION SUMMARY

Case Number: NR-2024-00567
Date: March 20, 2024
Operation: Operation Clean Streets
Location: Multiple locations, Little Rock, Arkansas

Lead Investigator: Detective Captain Maria Gonzalez
Task Force: DEA-LR Joint Task Force, Little Rock PD Narcotics Division

Summary:
Following a six-month investigation, the joint task force executed search
warrants at three locations in the Little Rock metropolitan area on March 20,
2024. The operation targeted a suspected drug distribution network operating
in the west Little Rock area.

Search Warrant Locations:
1. 1200 Washington Avenue — Primary stash house
2. 3400 Pine Street — Secondary distribution point
3. 78 Riverside Drive — Suspect residence

Seized Evidence:
- 2.5 kg cocaine (estimated street value $75,000)
- 1.2 kg methamphetamine (estimated street value $36,000)
- $42,000 in U.S. currency
- 3 firearms (2 handguns, 1 rifle)
- Drug packaging materials and digital scales
- Multiple cell phones and electronic devices

Arrests Made:
1. Carlos Mendez, age 38 — Possession with intent to distribute (Schedule II)
2. Anthony Brown, age 29 — Possession with intent to distribute
3. Jasmine Walker, age 26 — Conspiracy to distribute controlled substances

Outcome: All three suspects arraigned. Held without bail pending trial.
DEA continuing investigation into upstream supply chain.
""",
        "filename": "narcotics_investigation_littlerock.pdf"
    },
]


# ============================================================================
# STEP 1: PDF TEXT EXTRACTION
# ============================================================================

def extract_text_with_pymupdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz) — primary extractor."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        print(f"  [PyMuPDF] Opened PDF: {len(doc)} page(s)")
        full_text = ""
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and len(text.strip()) > 20:
                full_text += text + "\n"
        doc.close()
        return full_text.strip()
    except ImportError:
        print("  [PyMuPDF] fitz not installed, trying pdfplumber...")
        return None
    except Exception as e:
        print(f"  [PyMuPDF] Error: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    Uses PyMuPDF (fitz) as primary, pdfplumber as secondary,
    and OCR (pytesseract) as fallback for scanned pages.
    """
    # Try PyMuPDF first (recommended by assignment)
    full_text = extract_text_with_pymupdf(pdf_path)
    if full_text and len(full_text) > 50:
        return full_text

    # Try pdfplumber as secondary
    full_text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  [pdfplumber] Opened PDF: {len(pdf.pages)} page(s)")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    full_text += text + "\n"
                else:
                    # Try OCR fallback for scanned pages
                    print(f"    Page {i+1}: No text found, trying OCR...")
                    ocr_text = ocr_page(pdf_path, i)
                    if ocr_text:
                        full_text += ocr_text + "\n"
                    else:
                        print(f"    Page {i+1}: OCR also returned no text.")
    except Exception as e:
        print(f"  [pdfplumber] Error: {e}")
        # Try full OCR as last resort
        full_text = ocr_full_pdf(pdf_path)

    return full_text.strip()


def ocr_page_with_pymupdf(doc, page_number):
    """
    OCR a single PDF page by rendering it to an image via PyMuPDF's get_pixmap(),
    then running pytesseract. Does NOT require poppler/pdf2image.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Set tesseract path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        page = doc[page_number]
        # Render page at 300 DPI for good OCR quality
        mat = __import__('fitz').Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Run OCR
        text = pytesseract.image_to_string(img)
        return text.strip()
    except ImportError:
        return ""
    except Exception as e:
        print(f"    [OCR] Error on page {page_number + 1}: {e}")
        return ""


def ocr_page(pdf_path, page_number):
    """Attempt OCR on a specific PDF page using pytesseract."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Set tesseract path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        images = convert_from_path(pdf_path, first_page=page_number + 1,
                                   last_page=page_number + 1)
        if images:
            text = pytesseract.image_to_string(images[0])
            return text.strip()
    except ImportError:
        print("    [OCR] pdf2image or pytesseract not available.")
    except Exception as e:
        print(f"    [OCR] Error: {e}")
    return ""


def ocr_full_pdf(pdf_path):
    """Attempt full OCR on the entire PDF using PyMuPDF rendering."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        # Set tesseract path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        print(f"  [OCR] Running full OCR on PDF...")
        doc = fitz.open(pdf_path)
        texts = []
        mat = fitz.Matrix(300 / 72, 300 / 72)
        for i in range(doc.page_count):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img)
            if text.strip():
                texts.append(text.strip())
            if (i + 1) % 10 == 0:
                print(f"    [OCR] Processed {i + 1}/{doc.page_count} pages...")
        doc.close()
        return "\n".join(texts)
    except ImportError:
        print("  [OCR] pytesseract not installed.")
    except Exception as e:
        print(f"  [OCR] Full OCR error: {e}")
    return ""


# ============================================================================
# STEP 2: ENTITY EXTRACTION (spaCy NER)
# ============================================================================

def load_spacy_model():
    """Load the spaCy English NER model."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("[PDF] spaCy 'en_core_web_sm' model loaded.")
        return nlp
    except OSError:
        print("[PDF] Downloading spaCy model...")
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp


def extract_entities(nlp, text):
    """
    Extract structured entities from document text using spaCy NER.
    Returns dict with: date, location, officer, organizations, persons.
    """
    doc = nlp(text[:5000])  # Limit text length for NER

    entities = {
        "dates": [],
        "locations": [],
        "persons": [],
        "organizations": [],
    }

    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC", "FAC"):
            entities["locations"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)

    return entities


# ============================================================================
# STEP 3: STRUCTURED FIELD EXTRACTION
# ============================================================================

def extract_date(text, ner_dates):
    """Extract the most relevant date from the document."""
    # Try regex patterns for common date formats
    date_patterns = [
        r'Date(?:\s*of\s*Incident)?:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'Date(?:\s*Filed)?:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'Date:\s*(\d{1,2}/\d{1,2}/\d{4})',
        r'Date:\s*(\d{4}-\d{2}-\d{2})',
        r'Date:\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Fall back to NER dates
    if ner_dates:
        # Filter out generic dates like "today" or "this year"
        for d in ner_dates:
            if any(char.isdigit() for char in d) and len(d) > 4:
                return d

    return "Unknown"


def extract_location(text, ner_locations):
    """Extract the primary location from the document."""
    # Try regex for explicit location fields
    loc_patterns = [
        r'Location:\s*(.+?)(?:\n|$)',
        r'Address:\s*(.+?)(?:\n|$)',
        r'Scene:\s*(.+?)(?:\n|$)',
    ]

    for pattern in loc_patterns:
        match = re.search(pattern, text)
        if match:
            loc = match.group(1).replace('\n', ' ').strip()
            if len(loc) > 3 and len(loc) < 100:
                return loc

    # Fall back to NER locations
    if ner_locations:
        # Return the most specific location (longest one usually)
        return max(ner_locations, key=len).replace('\n', ' ').strip()

    return "Unknown"


def extract_officer(text, ner_persons):
    """Extract the reporting officer's name from the document."""
    # Try regex patterns
    officer_patterns = [
        r'(?:Reporting|Investigating|Requesting)\s+Officer:\s*(.+?)(?:\n|,|$)',
        r'(?:Officer|Detective|Sergeant|Lieutenant|Captain)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'Prepared by:\s*(.+?)(?:\n|$)',
        r'Lead Investigator:\s*(.+?)(?:\n|$)',
    ]

    for pattern in officer_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).replace('\n', ' ').strip()
            # Clean up title prefixes
            name = re.sub(r'^(Quartermaster\s+)?', '', name)
            if len(name) > 3 and len(name) < 60:
                return name

    # Fall back to first NER person
    if ner_persons:
        return ner_persons[0].replace('\n', ' ').strip()

    return "Unknown"


def extract_department(text, ner_orgs):
    """
    Extract the department name from the document.
    Tries regex patterns first, then falls back to NER organizations.
    """
    # Try regex patterns for department mentions
    dept_patterns = [
        r'Department:\s*(.+?)(?:\n|$)',
        r'([A-Z][\w\s]+(?:Police Department|Sheriff[\u2019\'s]*\s*Office|PD|County Sheriff))',
        r'([A-Z][\w\s]+(?:Law Enforcement|Public Safety))',
    ]

    for pattern in dept_patterns:
        match = re.search(pattern, text)
        if match:
            dept = match.group(1).replace('\n', ' ').strip()
            if len(dept) > 3 and len(dept) < 80:
                return dept

    # Fall back to NER organizations
    if ner_orgs:
        # Prefer org names that look like departments
        for org in ner_orgs:
            org_lower = org.lower()
            if any(kw in org_lower for kw in ['police', 'sheriff', 'department', 'law']):
                return org.replace('\n', ' ').strip()
        return ner_orgs[0].replace('\n', ' ').strip()

    return "Unknown"


def extract_program(text, incident_type):
    """
    Extract the program name from the document.
    For 1033-type documents, defaults to 'Law Enforcement Support'.
    """
    # Try regex patterns
    prog_patterns = [
        r'Program:\s*(.+?)(?:\n|$)',
        r'Program Name:\s*(.+?)(?:\n|$)',
        r'Subject:\s*(.+?)(?:\n|$)',
    ]

    for pattern in prog_patterns:
        match = re.search(pattern, text)
        if match:
            prog = match.group(1).replace('\n', ' ').strip()
            if len(prog) > 3 and len(prog) < 100:
                return prog

    # Default based on incident type
    program_map = {
        "1033 Training Proposal": "Law Enforcement Support",
        "Equipment Inventory": "Equipment Management",
        "Narcotics Investigation": "Narcotics Unit",
        "Arson / Fire Investigation": "Fire Investigation Unit",
    }

    return program_map.get(incident_type, "General Operations")


def classify_incident_type(text):
    """Classify the incident type based on keywords in the text.
    
    IMPORTANT: More specific types (Training Proposal, Equipment Inventory) are
    checked FIRST so that generic keywords like 'vehicle' in 'MRAP Vehicle' don't
    falsely trigger 'Traffic Accident'.
    """
    text_lower = text.lower()

    # Ordered list — more specific / niche types FIRST to avoid false matches
    incident_types = [
        ("1033 Training Proposal", [
            "training plan", "1033 program", "1033", "equipment request",
            "certification", "mrap", "mine resistant", "continuing education",
            "clest", "law enforcement support", "tactical", "instructor",
            "course:", "lesson title", "surplus", "defense logistics",
        ]),
        ("Equipment Inventory", [
            "inventory", "equipment audit", "quartermaster", "ammunition",
            "equipment list", "serial number", "property book",
        ]),
        ("Arson / Fire Investigation", [
            "arson", "fire investigation", "accelerant", "ignition",
            "engulfed in flames", "fire scene", "burn pattern",
        ]),
        ("Armed Robbery", [
            "robbery", "robbed", "brandished", "demanded cash", "armed robbery",
        ]),
        ("Theft / Burglary", [
            "theft", "stolen", "burglary", "breaking and entering",
            "forcibly entered", "items stolen",
        ]),
        ("Shooting", [
            "shooting", "gunshot", "shots fired", "shell casings",
            "firearm discharge",
        ]),
        ("Assault", [
            "assault", "fight", "altercation", "struck", "physical", "battery",
        ]),
        ("Traffic Accident", [
            "traffic accident", "collision", "red light", "intersection",
            "driver struck", "rear-ended", "head-on",
        ]),
        ("Public Disturbance", [
            "disturbance", "disorderly", "crowd", "noise", "dispersal",
        ]),
        ("Narcotics Investigation", [
            "narcotics", "drug", "cocaine", "methamphetamine",
            "controlled substance", "distribution",
        ]),
    ]

    best_type = "General Incident Report"
    best_score = 0

    for incident_type, keywords in incident_types:
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_type = incident_type

    return best_type


def generate_summary(text, incident_type, entities):
    """
    Generate a concise summary from the document text.
    Tries BART transformer summarization first, falls back to rule-based
    extraction of key details (suspect, outcome, damage, key phrases).
    """
    summary_parts = []

    # Add incident type
    summary_parts.append(incident_type)

    # Try BART summarization for rich summaries
    try:
        from transformers import pipeline as hf_pipeline
        summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn",
                                  device=-1)
        # Use first 1024 chars to stay within model limits
        chunk = text[:1024].strip()
        if len(chunk) > 100:
            result = summarizer(chunk, max_length=80, min_length=20,
                                do_sample=False)
            bart_summary = result[0]["summary_text"]
            summary_parts.append(bart_summary)
    except Exception:
        # BART not available — use rule-based fallback
        pass

    # Extract suspect description if present
    suspect_patterns = [
        r'Suspect\s*(?:Description)?\s*(?:\d)?:\s*(.+?)(?:\n\n|\n[A-Z])',
        r'Suspect:\s*(.+?)(?:\n\n|\n[A-Z])',
        r'Description of Suspect:\s*(.+?)(?:\n\n|\n[A-Z])',
    ]
    for pattern in suspect_patterns:
        suspect_match = re.search(pattern, text, re.DOTALL)
        if suspect_match:
            suspect_text = suspect_match.group(1).strip().replace('\n', ' ')
            if len(suspect_text) < 200:
                summary_parts.append(f"Suspect: {suspect_text}")
            break

    # Extract outcome if present
    outcome_match = re.search(r'Outcome:\s*(.+?)(?:\n|$)', text)
    if outcome_match:
        summary_parts.append(f"Outcome: {outcome_match.group(1).strip()}")

    # Extract damage/loss if present
    damage_match = re.search(r'(?:Estimated Damage|Total estimated loss|Damage)\s*:\s*(.+?)(?:\n|$)', text)
    if damage_match:
        summary_parts.append(f"Damage: {damage_match.group(1).strip()}")

    # Extract key detail from first meaningful line of text
    if len(summary_parts) <= 1:
        # Fallback: grab first non-header line as key detail
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
        for line in lines:
            if not line.startswith(('To:', 'From:', 'Date:', 'Ref:', 'COURSE:')):
                key_detail = line[:150]
                summary_parts.append(f"Detail: {key_detail}")
                break

    summary = ". ".join(summary_parts)

    # Truncate if too long
    if len(summary) > 350:
        summary = summary[:347] + "..."

    return summary


# ============================================================================
# STEP 4: LOAD DOCUMENTS
# ============================================================================

def split_pdf_into_sections(pdf_path):
    """
    Split a multi-section PDF into separate document sections.
    Uses PyMuPDF for digital text and OCR (pytesseract) for scanned pages.
    Detects boundaries by page gaps (empty pages between sections)
    and header patterns (To:, COURSE:, organization names).
    Returns list of dicts with 'text' and 'section_label'.
    """
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"  [Split] Scanning {total_pages} pages (digital text + OCR for scanned)...")

    sections = []
    current_section_pages = []
    current_text_parts = []
    digital_count = 0
    ocr_count = 0
    empty_count = 0

    for i in range(total_pages):
        page = doc[i]
        text = page.get_text().strip()

        # If fitz returns no/minimal text, this page is likely a scanned image → OCR it
        if not text or len(text) <= 20:
            ocr_text = ocr_page_with_pymupdf(doc, i)
            if ocr_text and len(ocr_text.strip()) > 20:
                text = ocr_text.strip()
                ocr_count += 1
                if (ocr_count % 10) == 0:
                    print(f"    [OCR] Processed {ocr_count} scanned pages so far "
                          f"(page {i+1}/{total_pages})...")
            else:
                # Truly empty page (blank separator, cover page, etc.)
                empty_count += 1
                continue
        else:
            digital_count += 1

        # Check if this page starts a new section
        is_new_section = False
        if current_text_parts:
            # Gap of empty pages = new section
            if current_section_pages and i - current_section_pages[-1] > 1:
                is_new_section = True
            # "To:" header at start = new section
            elif text.startswith("To:") or text.startswith("To :"):
                is_new_section = True
            # "To Whom It May Concern" in first 200 chars
            elif "To Whom It May Concern" in text[:200]:
                is_new_section = True
            # New organization name at start = new section
            elif any(text.startswith(org) for org in [
                "Lonoke", "Fort Smith", "Department of",
                "Sheriff", "Police Department", "Office of"
            ]):
                is_new_section = True
            # OCR'd pages often start with department headers (case-insensitive)
            elif any(kw in text[:300].lower() for kw in [
                "police department", "sheriff's office", "sheriff\u2019s office",
                "department of", "county sheriff", "state of arkansas",
                "law enforcement", "request for", "training plan",
                "1033 program", "course:", "date:",
                "standard operating procedure", "lesson plan",
                "policies and procedures", "mine resistant", "mrap vehicle",
            ]):
                is_new_section = True

        if is_new_section and current_text_parts:
            # Save current section
            full_text = "\n".join(current_text_parts)
            if len(full_text) > 50:  # Only save meaningful sections
                sections.append({
                    "text": full_text,
                    "section_label": f"Section_{len(sections)+1}_pages_{current_section_pages[0]+1}-{current_section_pages[-1]+1}"
                })
            current_text_parts = []
            current_section_pages = []

        current_text_parts.append(text)
        current_section_pages.append(i)

    # Save last section
    if current_text_parts:
        full_text = "\n".join(current_text_parts)
        if len(full_text) > 50:
            sections.append({
                "text": full_text,
                "section_label": f"Section_{len(sections)+1}_pages_{current_section_pages[0]+1}-{current_section_pages[-1]+1}"
            })

    doc.close()
    print(f"  [Split] Results: {digital_count} digital pages, {ocr_count} OCR'd pages, "
          f"{empty_count} blank pages → {len(sections)} sections")
    return sections


def load_pdf_documents():
    """
    Load PDFs from data directory or fall back to sample reports.
    Splits multi-section PDFs into separate case records.
    Returns list of dicts with 'text' and 'filename' keys.
    """
    documents = []

    # Try loading real PDF files
    if os.path.exists(DATA_DIR):
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"[PDF] Found {len(pdf_files)} PDF file(s) in data directory.")
            for pdf_file in sorted(pdf_files):
                pdf_path = os.path.join(DATA_DIR, pdf_file)
                print(f"\n  Processing: {pdf_file}")

                # Try splitting into sections first
                sections = split_pdf_into_sections(pdf_path)

                if sections and len(sections) > 1:
                    print(f"  [OK] Found {len(sections)} separate report sections")
                    for sec in sections:
                        if sec["text"] and len(sec["text"]) > 50:
                            documents.append({
                                "text": sec["text"],
                                "filename": f"{pdf_file} ({sec['section_label']})"
                            })
                            print(f"    -> {sec['section_label']}: {len(sec['text'])} chars")
                elif sections and len(sections) == 1:
                    # Single section — use as-is
                    text = sections[0]["text"]
                    if text and len(text) > 50:
                        documents.append({
                            "text": text,
                            "filename": pdf_file
                        })
                        print(f"  [OK] Extracted {len(text)} characters (single report)")
                else:
                    # Fallback to full-text extraction
                    text = extract_text_from_pdf(pdf_path)
                    if text and len(text) > 50:
                        documents.append({
                            "text": text,
                            "filename": pdf_file
                        })
                        print(f"  [OK] Extracted {len(text)} characters")
                    else:
                        print(f"  [!] Could not extract meaningful text from {pdf_file}")

    if documents:
        return documents

    # Fall back to sample reports
    print("[PDF] No PDF files found. Using built-in sample police reports.")
    print(f"[PDF] {len(SAMPLE_REPORTS)} sample reports loaded.")
    return SAMPLE_REPORTS


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pdf_pipeline():
    """
    Execute the full PDF/Document analysis pipeline:
    1. Load PDF documents (real or sample)
    2. Extract entities with spaCy NER
    3. Classify incident types
    4. Extract structured fields
    5. Export to CSV
    """
    print("=" * 70)
    print("PDF/DOCUMENT PIPELINE — Student 2 (Neha Reddy Poreddy)")
    print("Multimodal Crime / Incident Report Analyzer")
    print("=" * 70)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    documents = load_pdf_documents()

    if not documents:
        print("[PDF] ERROR: No documents available. Exiting.")
        sys.exit(1)

    # Step 2: Load NER model
    print("\n[Step 2] Loading spaCy NER model...")
    nlp = load_spacy_model()

    # Step 3: Process each document
    print(f"\n[Step 3] Processing {len(documents)} documents...")
    results = []

    for idx, doc in enumerate(documents):
        report_id = f"RPT_{(idx + 1):03d}"
        text = doc["text"]
        filename = doc["filename"]

        print(f"\n  [{report_id}] {filename}")

        # Extract entities using spaCy
        entities = extract_entities(nlp, text)

        # Classify incident type
        incident_type = classify_incident_type(text)

        # Extract structured fields
        date = extract_date(text, entities["dates"])
        location = extract_location(text, entities["locations"])
        officer = extract_officer(text, entities["persons"])
        department = extract_department(text, list(entities["organizations"]))
        program = extract_program(text, incident_type)

        # Generate summary
        summary = generate_summary(text, incident_type, entities)

        # Determine document type based on content
        doc_type = "Training Proposal"
        text_lower = text.lower()
        if "standard operating procedure" in text_lower or "sop" in text_lower:
            doc_type = "SOP"
        elif "lesson plan" in text_lower or "course:" in text_lower:
            doc_type = "Lesson Plan"
        elif "policies and procedures" in text_lower:
            doc_type = "Policy Document"
        elif "request for" in text_lower:
            doc_type = "Equipment Request"
        elif "inventory" in text_lower:
            doc_type = "Inventory Report"

        # Extract key detail (first meaningful sentence from summary or text)
        key_detail = summary.split(".")[0].strip() if summary else ""
        if not key_detail or key_detail == "Unknown":
            sentences = [s.strip() for s in text[:500].split(".") if len(s.strip()) > 20]
            key_detail = sentences[0] if sentences else "Document section"

        results.append({
            "Report_ID": report_id,
            "Department": department,
            "Incident_Type": incident_type,
            "Doc_Type": doc_type,
            "Date": date,
            "Location": location,
            "Program": program,
            "Officer": officer,
            "Summary": summary,
            "Key_Detail": key_detail,
        })

        print(f"    Type: {incident_type} | Doc: {doc_type}")
        print(f"    Dept: {department} | Program: {program}")
        print(f"    Date: {date} | Location: {location}")
        print(f"    Officer: {officer}")
        print(f"    Summary: {summary[:80]}...")

    # Step 4: Export to CSV
    print(f"\n[Step 4] Exporting results to '{OUTPUT_FILE}'...")
    df = pd.DataFrame(results)

    # Column order per assignment spec
    df = df[["Report_ID", "Department", "Incident_Type", "Doc_Type", "Date",
             "Location", "Program", "Officer", "Summary", "Key_Detail"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"[PDF] Pipeline complete! Output saved to: {OUTPUT_FILE}")
    print(f"[PDF] Total records: {len(df)}")
    print(f"{'=' * 70}")

    # Display summary
    print("\n--- Output Preview ---")
    print(df.to_string(index=False, max_colwidth=50))

    return df


if __name__ == "__main__":
    run_pdf_pipeline()
