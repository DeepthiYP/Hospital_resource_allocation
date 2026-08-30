import os
import pandas as pd
from datetime import datetime

from services.occupancy import load_bed_data


WAITING_LIST_FILE = "dataset/waiting_list.csv"


# =========================================================
# NORMALIZE VALUES
# =========================================================

def normalize(value):
    """
    Converts values into a consistent lowercase format.
    This prevents problems such as:
    ICU vs icu
    Yes vs yes
    Available vs available
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# =========================================================
# MATCH SCORE
# =========================================================

def calculate_match_score(bed, requirements):
    """
    Calculates the percentage of required resources
    that match the bed.

    Ward is always required.

    Optional resources are counted only when requested.
    """

    total = 0
    matched = 0

    # -----------------------------
    # WARD
    # -----------------------------

    total += 1

    if normalize(bed["ward_type"]) == normalize(
        requirements["ward_type"]
    ):
        matched += 1

    # -----------------------------
    # OXYGEN
    # -----------------------------

    if requirements.get("oxygen", False):

        total += 1

        if normalize(bed["oxygen"]) == "yes":
            matched += 1

    # -----------------------------
    # VENTILATOR
    # -----------------------------

    if requirements.get("ventilator", False):

        total += 1

        if normalize(bed["ventilator"]) == "yes":
            matched += 1

    # -----------------------------
    # ISOLATION
    # -----------------------------

    if requirements.get("isolation", False):

        total += 1

        if normalize(bed["isolation"]) == "yes":
            matched += 1

    if total == 0:
        return 0

    return round(
        (matched / total) * 100
    )


# =========================================================
# STRICT BED MATCH
# =========================================================

def bed_matches_requirements(bed, requirements):
    """
    A bed is suitable ONLY if every required
    condition is satisfied.

    IMPORTANT:
    Ward type is mandatory.

    Therefore:

    ICU request != Isolation bed

    even if the Isolation bed has oxygen,
    ventilator and isolation.
    """

    # -----------------------------
    # WARD
    # -----------------------------

    if normalize(bed["ward_type"]) != normalize(
        requirements["ward_type"]
    ):
        return False

    # -----------------------------
    # OXYGEN
    # -----------------------------

    if requirements.get("oxygen", False):

        if normalize(bed["oxygen"]) != "yes":
            return False

    # -----------------------------
    # VENTILATOR
    # -----------------------------

    if requirements.get("ventilator", False):

        if normalize(bed["ventilator"]) != "yes":
            return False

    # -----------------------------
    # ISOLATION
    # -----------------------------

    if requirements.get("isolation", False):

        if normalize(bed["isolation"]) != "yes":
            return False

    return True


# =========================================================
# CURRENT AVAILABLE BEDS
# =========================================================

def find_current_beds(
    beds,
    requirements
):
    """
    Finds beds that are currently Available
    AND satisfy every requirement.
    """

    if beds.empty:
        return beds.copy()

    available = beds[
        beds["status"].apply(normalize) == "available"
    ].copy()

    if available.empty:
        return available

    matching = available[
        available.apply(
            lambda row:
            bed_matches_requirements(
                row,
                requirements
            ),
            axis=1
        )
    ].copy()

    return matching


# =========================================================
# FUTURE BED CHECK
# =========================================================

def find_future_beds(
    beds,
    admissions,
    requirements
):
    """
    Finds occupied beds that:

    1. Match the required ward
    2. Match oxygen requirement
    3. Match ventilator requirement
    4. Match isolation requirement
    5. Have a valid expected discharge date

    The future bed becomes available after the
    current patient's expected discharge.
    """

    if beds.empty or admissions.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # ONLY OCCUPIED BEDS
    # -----------------------------------------

    occupied = beds[
        beds["status"].apply(normalize) == "occupied"
    ].copy()

    if occupied.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # MATCH REQUIREMENTS FIRST
    # -----------------------------------------

    occupied = occupied[
        occupied.apply(
            lambda row:
            bed_matches_requirements(
                row,
                requirements
            ),
            axis=1
        )
    ].copy()

    if occupied.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # ONLY ACTIVE ADMISSIONS
    # -----------------------------------------

    active_admissions = admissions.copy()

    if "status" in active_admissions.columns:

        active_admissions = active_admissions[
            active_admissions["status"]
            .apply(normalize)
            == "active"
        ].copy()

    if active_admissions.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # CONVERT DISCHARGE DATE
    # -----------------------------------------

    active_admissions[
        "expected_discharge_date"
    ] = pd.to_datetime(
        active_admissions[
            "expected_discharge_date"
        ],
        errors="coerce"
    )

    active_admissions = active_admissions[
        active_admissions[
            "expected_discharge_date"
        ].notna()
    ].copy()

    if active_admissions.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # MERGE BED + ADMISSION
    # -----------------------------------------

    future = occupied.merge(
        active_admissions[
            [
                "bed_id",
                "expected_discharge_date"
            ]
        ],
        on="bed_id",
        how="inner"
    )

    if future.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # TODAY
    # -----------------------------------------

    today = pd.Timestamp.now().normalize()

    # -----------------------------------------
    # DAYS UNTIL BED IS FREE
    # -----------------------------------------

    future[
        "days_until_available"
    ] = (
        future["expected_discharge_date"]
        - today
    ).dt.days

    # Ignore beds whose discharge date has passed.
    future = future[
        future["days_until_available"] >= 0
    ].copy()

    return future


# =========================================================
# WAITING LIST
# =========================================================

def add_to_waiting_list(
    requirements,
    reason
):
    """
    Adds patient request to waiting_list.csv.
    """

    directory = os.path.dirname(
        WAITING_LIST_FILE
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    try:

        waiting = pd.read_csv(
            WAITING_LIST_FILE
        )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError
    ):

        waiting = pd.DataFrame(
            columns=[
                "request_id",
                "patient_id",
                "request_date",
                "ward_type",
                "stay_days",
                "oxygen",
                "ventilator",
                "isolation",
                "status",
                "reason"
            ]
        )

    # -----------------------------------------
    # REQUEST ID
    # -----------------------------------------

    request_number = len(waiting) + 1

    request_id = (
        f"REQ{request_number:03d}"
    )

    # -----------------------------------------
    # PATIENT ID
    # -----------------------------------------

    patient_id = requirements.get(
        "patient_id",
        f"P{request_number + 6:03d}"
    )

    # -----------------------------------------
    # NEW REQUEST
    # -----------------------------------------

    new_request = {

        "request_id":
            request_id,

        "patient_id":
            patient_id,

        "request_date":
            datetime.now().strftime(
                "%Y-%m-%d"
            ),

        "ward_type":
            requirements["ward_type"],

        "stay_days":
            requirements["stay_days"],

        "oxygen":
            "Yes"
            if requirements.get("oxygen", False)
            else "No",

        "ventilator":
            "Yes"
            if requirements.get("ventilator", False)
            else "No",

        "isolation":
            "Yes"
            if requirements.get("isolation", False)
            else "No",

        "status":
            "Waiting",

        "reason":
            reason
    }

    # -----------------------------------------
    # ADD
    # -----------------------------------------

    waiting.loc[
        len(waiting)
    ] = new_request

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    waiting.to_csv(
        WAITING_LIST_FILE,
        index=False
    )

    return new_request


# =========================================================
# MAIN ALLOCATION FUNCTION
# =========================================================

def allocate_bed(requirements):

    beds, admissions, _ = load_bed_data()

    # =====================================================
    # CLEAN REQUIREMENTS
    # =====================================================

    requirements["ward_type"] = str(
        requirements.get(
            "ward_type",
            "General"
        )
    ).strip()

    stay_days = requirements.get(
        "stay_days",
        1
    )

    try:
        stay_days = int(stay_days)

    except (
        ValueError,
        TypeError
    ):
        stay_days = 1

    if stay_days < 1:
        stay_days = 1

    requirements["stay_days"] = stay_days

    # =====================================================
    # DATE CALCULATIONS
    # =====================================================

    today = pd.Timestamp.now().normalize()

    expected_release_date = (
        today
        + pd.Timedelta(
            days=stay_days
        )
    )

    # =====================================================
    # 1. CHECK CURRENTLY AVAILABLE BEDS
    # =====================================================

    current = find_current_beds(
        beds,
        requirements
    )

    if not current.empty:

        current = current.copy()

        # Calculate scores
        current["score"] = current.apply(
            lambda row:
            calculate_match_score(
                row,
                requirements
            ),
            axis=1
        )

        # Highest score first
        current = current.sort_values(
            by="score",
            ascending=False
        )

        best = current.iloc[0]

        return {

            "status":
                "AVAILABLE",

            "bed_id":
                str(best["bed_id"]),

            "room_id":
                str(best["room_id"]),

            "ward_type":
                str(best["ward_type"]),

            "bed_type":
                str(best["bed_type"]),

            "match_score":
                int(best["score"]),

            "stay_days":
                stay_days,

            "allocation_start":
                str(today.date()),

            "expected_release_date":
                str(
                    expected_release_date.date()
                ),

            "message":
                (
                    "Suitable bed is currently "
                    "available and can accommodate "
                    f"the requested {stay_days}-day stay."
                )
        }

    # =====================================================
    # 2. CHECK FUTURE BEDS
    # =====================================================

    future = find_future_beds(
        beds,
        admissions,
        requirements
    )

    if not future.empty:

        future = future.sort_values(
            by="expected_discharge_date"
        )

        best = future.iloc[0]

        available_date = pd.Timestamp(
            best[
                "expected_discharge_date"
            ]
        )

        future_release_date = (
            available_date
            + pd.Timedelta(
                days=stay_days
            )
        )

        return {

            "status":
                "FUTURE_AVAILABLE",

            "bed_id":
                str(best["bed_id"]),

            "room_id":
                str(best["room_id"]),

            "ward_type":
                str(best["ward_type"]),

            "bed_type":
                str(best["bed_type"]),

            "stay_days":
                stay_days,

            "available_date":
                str(
                    available_date.date()
                ),

            "expected_release_date":
                str(
                    future_release_date.date()
                ),

            "days_until_available":
                int(
                    best[
                        "days_until_available"
                    ]
                ),

            "match_score":
                100,

            "message":
                (
                    "No suitable bed is available "
                    "right now. A matching bed is "
                    "expected to become available on "
                    f"{available_date.date()}."
                )
        }

    # =====================================================
    # 3. WAITING LIST
    # =====================================================

    reason = (
        "No suitable bed is currently available "
        "and no matching future bed was found. "
        "The request requires an exact match for "
        f"ward={requirements['ward_type']}, "
        f"oxygen={requirements.get('oxygen', False)}, "
        f"ventilator={requirements.get('ventilator', False)}, "
        f"isolation={requirements.get('isolation', False)}, "
        f"and stay={stay_days} days."
    )

    waiting_request = add_to_waiting_list(
        requirements,
        reason
    )

    return {

        "status":
            "WAITING_LIST",

        "match_score":
            0,

        "stay_days":
            stay_days,

        "waiting_request_id":
            waiting_request["request_id"],

        "message":
            (
                "No suitable bed is currently "
                "available and no matching future "
                "bed was found. The patient has "
                "been added to the waiting list."
            )
    }