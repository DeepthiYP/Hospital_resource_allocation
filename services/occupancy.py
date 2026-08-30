import pandas as pd
from datetime import datetime


DATA_PATH = "dataset/"


def load_bed_data():

    beds = pd.read_csv(
        DATA_PATH + "beds.csv"
    )

    admissions = pd.read_csv(
        DATA_PATH + "admissions.csv"
    )

    patients = pd.read_csv(
        DATA_PATH + "patients.csv"
    )

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"]
    )

    admissions["expected_discharge_date"] = pd.to_datetime(
        admissions["expected_discharge_date"]
    )

    return beds, admissions, patients


def get_occupancy_summary():

    beds, admissions, patients = load_bed_data()

    total_beds = len(beds)

    occupied_beds = len(
        beds[beds["status"] == "Occupied"]
    )

    available_beds = len(
        beds[beds["status"] == "Available"]
    )

    occupancy_rate = (
        occupied_beds / total_beds
    ) * 100 if total_beds else 0

    return {
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
        "occupancy_rate": round(
            occupancy_rate, 2
        )
    }


def get_occupied_beds():

    beds, admissions, patients = load_bed_data()

    occupied = beds[
        beds["status"] == "Occupied"
    ]

    result = occupied.merge(
        admissions,
        on="bed_id",
        how="left"
    )

    result = result.merge(
        patients,
        on="patient_id",
        how="left"
    )

    today = pd.Timestamp.now().normalize()

    result["occupied_days"] = (
        today - result["admission_date"]
    ).dt.days

    result["days_remaining"] = (
        result["expected_discharge_date"]
        - today
    ).dt.days

    return result.fillna("")


def get_ward_summary():

    beds, _, _ = load_bed_data()

    summary = beds.groupby(
        "ward_type"
    ).agg(
        total_beds=("bed_id", "count"),
        occupied_beds=(
            "status",
            lambda x:
            (x == "Occupied").sum()
        ),
        available_beds=(
            "status",
            lambda x:
            (x == "Available").sum()
        )
    )

    return summary.reset_index()