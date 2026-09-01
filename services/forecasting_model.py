import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model


MODEL_PATH = "models/occupancy_bilstm.keras"
CONFIG_PATH = "models/forecast_config.json"


def load_forecast_model():

    model = load_model(
        MODEL_PATH
    )

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = json.load(f)

    return model, config


def build_historical_occupancy():

    beds = pd.read_csv(
        "dataset/beds.csv"
    )

    admissions = pd.read_csv(
        "dataset/admissions.csv"
    )


    admissions[
        "admission_date"
    ] = pd.to_datetime(
        admissions[
            "admission_date"
        ]
    )


    admissions[
        "expected_discharge_date"
    ] = pd.to_datetime(
        admissions[
            "expected_discharge_date"
        ],
        errors="coerce"
    )


    admissions[
        "actual_discharge_date"
    ] = pd.to_datetime(
        admissions[
            "actual_discharge_date"
        ],
        errors="coerce"
    )


    # -------------------------------------------------
    # Connect admissions with ward
    # -------------------------------------------------

    admissions = admissions.merge(

        beds[
            [
                "bed_id",
                "ward_type"
            ]
        ],

        on="bed_id",

        how="left"
    )


    # -------------------------------------------------
    # Determine discharge date
    # -------------------------------------------------

    admissions[
        "discharge_date"
    ] = admissions[
        "actual_discharge_date"
    ].fillna(
        admissions[
            "expected_discharge_date"
        ]
    )


    admissions = admissions.dropna(
        subset=[
            "admission_date",
            "discharge_date",
            "ward_type"
        ]
    )


    # -------------------------------------------------
    # Date range
    # -------------------------------------------------

    start_date = admissions[
        "admission_date"
    ].min()

    end_date = max(

        admissions[
            "discharge_date"
        ].max(),

        pd.Timestamp.today()
    )


    dates = pd.date_range(
        start_date,
        end_date,
        freq="D"
    )


    wards = sorted(
        beds[
            "ward_type"
        ].dropna().unique()
    )


    records = []


    for date in dates:

        for ward in wards:

            active = admissions[

                (admissions[
                    "ward_type"
                ] == ward)

                &

                (
                    admissions[
                        "admission_date"
                    ] <= date
                )

                &

                (
                    admissions[
                        "discharge_date"
                    ] > date
                )

            ]

            occupancy = len(
                active
            )


            records.append({

                "date":
                    date,

                "ward_type":
                    ward,

                "occupancy":
                    occupancy

            })


    history = pd.DataFrame(
        records
    )


    return history


def forecast_occupancy(
    days=7
):

    model, config = (
        load_forecast_model()
    )


    history = (
        build_historical_occupancy()
    )


    sequence_length = int(
        config[
            "sequence_length"
        ]
    )


    wards = config[
        "wards"
    ]


    results = []


    for ward in wards:

        ward_data = history[
            history[
                "ward_type"
            ] == ward
        ].sort_values(
            "date"
        )


        values = ward_data[
            "occupancy"
        ].values.astype(
            "float32"
        )


        if len(values) < sequence_length:

            raise ValueError(

                f"Not enough historical "
                f"data for {ward}. "
                f"Required: "
                f"{sequence_length} days."
            )


        sequence = values[
            -sequence_length:
        ].reshape(
            1,
            sequence_length,
            1
        )


        predictions = []


        for _ in range(days):

            prediction = model.predict(
                sequence,
                verbose=0
            )


            value = max(
                0,
                float(
                    prediction[0][0]
                )
            )


            predictions.append(
                round(value, 2)
            )


            sequence = np.concatenate(

                [
                    sequence[:, 1:, :],

                    np.array(
                        [
                            [
                                [
                                    value
                                ]
                            ]
                        ],
                        dtype="float32"
                    )
                ],

                axis=1
            )


        for index, value in enumerate(
            predictions
        ):

            results.append({

                "day":
                    index + 1,

                "ward_type":
                    ward,

                "predicted_occupancy":
                    value

            })


    return results