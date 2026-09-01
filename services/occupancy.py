import os
import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

BEDS_FILE = os.path.join(
    DATASET_DIR,
    "beds.csv"
)

ADMISSIONS_FILE = os.path.join(
    DATASET_DIR,
    "admissions.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_bed_data():

    beds = pd.read_csv(
        BEDS_FILE
    )

    admissions = pd.read_csv(
        ADMISSIONS_FILE
    )

    rooms_file = os.path.join(
        DATASET_DIR,
        "rooms.csv"
    )

    if os.path.exists(rooms_file):
        rooms = pd.read_csv(
            rooms_file
        )
    else:
        rooms = pd.DataFrame()

    return beds, admissions, rooms


# =========================================================
# OCCUPANCY SUMMARY
# =========================================================

def get_occupancy_summary():

    beds, admissions, rooms = load_bed_data()

    total_beds = len(beds)

    occupied_beds = len(
        beds[
            beds["status"].astype(str).str.lower()
            == "occupied"
        ]
    )

    available_beds = len(
        beds[
            beds["status"].astype(str).str.lower()
            == "available"
        ]
    )

    if total_beds > 0:

        occupancy_rate = (
            occupied_beds /
            total_beds
        ) * 100

    else:

        occupancy_rate = 0

    return {

        "total_beds":
            total_beds,

        "occupied_beds":
            occupied_beds,

        "available_beds":
            available_beds,

        "occupancy_rate":
            round(
                occupancy_rate,
                1
            )
    }


# =========================================================
# OCCUPIED BEDS
# =========================================================

def get_occupied_beds():

    beds, admissions, rooms = load_bed_data()

    occupied = beds[
        beds["status"].astype(str).str.lower()
        == "occupied"
    ].copy()

    if occupied.empty:

        return occupied

    occupied = occupied.merge(

        admissions[
            [
                "patient_id",
                "bed_id",
                "admission_date",
                "expected_discharge_date",
                "status"
            ]
        ],

        on="bed_id",

        how="left",

        suffixes=(
            "",
            "_admission"
        )
    )

    occupied[
        "admission_date"
    ] = pd.to_datetime(
        occupied[
            "admission_date"
        ],
        errors="coerce"
    )

    occupied[
        "expected_discharge_date"
    ] = pd.to_datetime(
        occupied[
            "expected_discharge_date"
        ],
        errors="coerce"
    )

    today = pd.Timestamp.now().normalize()

    occupied[
        "days_occupied"
    ] = (
        today
        - occupied["admission_date"]
    ).dt.days

    occupied[
        "days_remaining"
    ] = (
        occupied[
            "expected_discharge_date"
        ]
        - today
    ).dt.days

    occupied[
        "days_occupied"
    ] = occupied[
        "days_occupied"
    ].fillna(0).clip(lower=0)

    occupied[
        "days_remaining"
    ] = occupied[
        "days_remaining"
    ].fillna(0).clip(lower=0)

    return occupied


# =========================================================
# WARD SUMMARY
# =========================================================

def get_ward_summary():

    beds, admissions, rooms = load_bed_data()

    result = []

    for ward in sorted(
        beds["ward_type"].dropna().unique()
    ):

        ward_beds = beds[
            beds["ward_type"] == ward
        ]

        total = len(
            ward_beds
        )

        occupied = len(
            ward_beds[
                ward_beds["status"]
                .astype(str)
                .str.lower()
                == "occupied"
            ]
        )

        available = len(
            ward_beds[
                ward_beds["status"]
                .astype(str)
                .str.lower()
                == "available"
            ]
        )

        occupancy = (
            occupied / total * 100
            if total > 0
            else 0
        )

        result.append({

            "ward_type":
                ward,

            "total":
                total,

            "occupied":
                occupied,

            "available":
                available,

            "occupancy":
                round(
                    occupancy,
                    1
                )
        })

    return pd.DataFrame(
        result
    )


# =========================================================
# DISCHARGE TREND
# =========================================================

def get_discharge_trend():

    beds, admissions, rooms = load_bed_data()

    if admissions.empty:

        return []

    admissions = admissions.copy()

    admissions[
        "expected_discharge_date"
    ] = pd.to_datetime(
        admissions[
            "expected_discharge_date"
        ],
        errors="coerce"
    )

    admissions = admissions.dropna(
        subset=[
            "expected_discharge_date"
        ]
    )

    if admissions.empty:

        return []

    today = pd.Timestamp.now().normalize()

    future = admissions[
        admissions[
            "expected_discharge_date"
        ] >= today
    ].copy()

    if future.empty:

        return []

    grouped = (
        future
        .groupby(
            "expected_discharge_date"
        )
        .size()
        .reset_index(
            name="discharges"
        )
        .sort_values(
            "expected_discharge_date"
        )
    )

    result = []

    for _, row in grouped.iterrows():

        result.append({

            "date":
                row[
                    "expected_discharge_date"
                ].strftime(
                    "%d %b"
                ),

            "discharges":
                int(
                    row["discharges"]
                )
        })

    return result