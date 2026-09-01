import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# SETTINGS
# =========================================================

SEQUENCE_LENGTH = 14

EPOCHS = 40

BATCH_SIZE = 16

MODEL_PATH = (
    "models/occupancy_bilstm.keras"
)

CONFIG_PATH = (
    "models/forecast_config.json"
)


os.makedirs(
    "models",
    exist_ok=True
)


# =========================================================
# BUILD HISTORICAL DATA
# =========================================================

from services.forecasting_model import (
    build_historical_occupancy
)


history = (
    build_historical_occupancy()
)


wards = sorted(
    history[
        "ward_type"
    ].unique()
)


print(
    "\nWards found:"
)

print(
    wards
)


# =========================================================
# CREATE TRAINING SEQUENCES
# =========================================================

X = []

y = []


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


    if len(values) <= SEQUENCE_LENGTH:

        print(
            f"Skipping {ward}: "
            f"not enough history."
        )

        continue


    # -----------------------------------------------------
    # Normalize separately for each ward
    # -----------------------------------------------------

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        values.reshape(-1, 1)
    ).flatten()


    for i in range(

        SEQUENCE_LENGTH,
        len(scaled)

    ):

        X.append(

            scaled[
                i - SEQUENCE_LENGTH:i
            ].reshape(
                SEQUENCE_LENGTH,
                1
            )
        )


        y.append(
            scaled[i]
        )


# =========================================================
# CHECK DATA
# =========================================================

X = np.array(X)

y = np.array(y)


print(
    "\nTraining samples:",
    len(X)
)


if len(X) < 20:

    raise ValueError(

        "Not enough historical "
        "time-series samples. "
        "Add more historical "
        "admission/discharge data."
    )


# =========================================================
# TRAIN / VALIDATION
# =========================================================

split = int(
    len(X) * 0.8
)


X_train = X[:split]

X_val = X[split:]


y_train = y[:split]

y_val = y[split:]


# =========================================================
# MODEL
# =========================================================

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            1
        )
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    Dropout(0.2),

    Bidirectional(
        LSTM(
            32
        )
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="linear"
    )

])


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=[
        "mae"
    ]
)


# =========================================================
# EARLY STOPPING
# =========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True
)


# =========================================================
# TRAIN
# =========================================================

model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop
    ],

    verbose=1
)


# =========================================================
# SAVE
# =========================================================

model.save(
    MODEL_PATH
)


config = {

    "sequence_length":
        SEQUENCE_LENGTH,

    "wards":
        wards

}


with open(
    CONFIG_PATH,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


print(
    "\n================================"
)

print(
    "OCCUPANCY BiLSTM TRAINED"
)

print(
    "================================"
)

print(
    "Model:",
    MODEL_PATH
)