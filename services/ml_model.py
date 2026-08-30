import re


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    return " ".join(
        str(text).replace("\n", " ").split()
    )


def find_value(text, patterns):
    """
    Finds a value after one of the supplied labels.
    """

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    return None


def contains_required(text, words):
    """
    Returns True when the document contains
    a phrase indicating that a resource is required.
    """

    text = text.lower()

    for word in words:

        if word in text:
            return True

    return False


# ==========================================
# WARD EXTRACTION
# ==========================================

def extract_ward(text):

    # First look for explicit ward labels.

    match = re.search(
        r"ward\s*(?:required|type)?\s*[:\-]?\s*"
        r"(icu|general|isolation|private)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip().title()

    # Fallback keyword detection.

    text_lower = text.lower()

    if "icu" in text_lower:
        return "ICU"

    if "isolation ward" in text_lower:
        return "Isolation"

    if "private ward" in text_lower:
        return "Private"

    return "General"


# ==========================================
# OXYGEN
# ==========================================

def extract_oxygen(text):

    text_lower = text.lower()

    # Explicit negative statements first.

    negative_patterns = [
        "oxygen support: not required",
        "oxygen: no",
        "oxygen required: no",
        "oxygen support no"
    ]

    for phrase in negative_patterns:

        if phrase in text_lower:
            return False

    positive_patterns = [
        "oxygen support: required",
        "oxygen: yes",
        "oxygen required: yes",
        "oxygen support required",
        "oxygen is required",
        "oxygen required"
    ]

    for phrase in positive_patterns:

        if phrase in text_lower:
            return True

    return False


# ==========================================
# VENTILATOR
# ==========================================

def extract_ventilator(text):

    text_lower = text.lower()

    negative_patterns = [
        "ventilator support: not required",
        "ventilator: no",
        "ventilator required: no",
        "ventilator support no"
    ]

    for phrase in negative_patterns:

        if phrase in text_lower:
            return False

    positive_patterns = [
        "ventilator support: required",
        "ventilator: yes",
        "ventilator required: yes",
        "ventilator support required",
        "ventilator is required",
        "ventilator required"
    ]

    for phrase in positive_patterns:

        if phrase in text_lower:
            return True

    return False


# ==========================================
# ISOLATION
# ==========================================

def extract_isolation(text):

    text_lower = text.lower()

    negative_patterns = [
        "isolation: no",
        "isolation required: no",
        "isolation support: no",
        "isolation: not required",
        "isolation not required"
    ]

    for phrase in negative_patterns:

        if phrase in text_lower:
            return False

    positive_patterns = [
        "isolation: yes",
        "isolation required: yes",
        "isolation support: yes",
        "isolation: required",
        "isolation required",
        "isolation is required",
        "requires isolation"
    ]

    for phrase in positive_patterns:

        if phrase in text_lower:
            return True

    return False


# ==========================================
# STAY DURATION
# ==========================================

def extract_stay_days(text):

    patterns = [

        r"expected\s+length\s+of\s+stay\s*[:\-]?\s*(\d+)\s*days?",

        r"length\s+of\s+stay\s*[:\-]?\s*(\d+)\s*days?",

        r"expected\s+stay\s*[:\-]?\s*(\d+)\s*days?",

        r"stay\s+duration\s*[:\-]?\s*(\d+)\s*days?",

        r"stay\s*[:\-]?\s*(\d+)\s*days?",

        r"(\d+)\s*days?\s*(?:of)?\s*(?:hospital|inpatient)\s*stay"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            days = int(match.group(1))

            if days >= 1:
                return days

    return 1


# ==========================================
# PATIENT ID
# ==========================================

def extract_patient_id(text):

    match = re.search(
        r"patient\s*(?:id|ID)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return "P999"


# ==========================================
# MAIN DOCUMENT ANALYSIS
# ==========================================

def predict_requirements(text):

    text = clean_text(text)

    requirements = {

        "patient_id":
            extract_patient_id(text),

        "ward_type":
            extract_ward(text),

        "stay_days":
            extract_stay_days(text),

        "oxygen":
            extract_oxygen(text),

        "ventilator":
            extract_ventilator(text),

        "isolation":
            extract_isolation(text)
    }

    print("\nMODEL/DOCUMENT ANALYSIS")
    print(requirements)

    return requirements