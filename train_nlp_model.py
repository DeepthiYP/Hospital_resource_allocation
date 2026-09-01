import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "dataset/medical_documents.csv"
MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "bilstm_nlp_model.keras"
)

TOKENIZER_FILE = os.path.join(
    MODEL_DIR,
    "nlp_tokenizer.pkl"
)

LABEL_ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "nlp_label_info.pkl"
)

MAX_WORDS = 5000
MAX_SEQUENCE_LENGTH = 50

EPOCHS = 30
BATCH_SIZE = 4


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading medical documents dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "document",
    "ward_type",
    "oxygen",
    "ventilator",
    "isolation",
    "stay_days"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in medical_documents.csv: "
        f"{missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()

df["document"] = (
    df["document"]
    .fillna("")
    .astype(str)
)

df["ward_type"] = (
    df["ward_type"]
    .fillna("General")
    .astype(str)
    .str.strip()
)

df["oxygen"] = (
    df["oxygen"]
    .fillna("No")
    .astype(str)
    .str.strip()
    .str.capitalize()
)

df["ventilator"] = (
    df["ventilator"]
    .fillna("No")
    .astype(str)
    .str.strip()
    .str.capitalize()
)

df["isolation"] = (
    df["isolation"]
    .fillna("No")
    .astype(str)
    .str.strip()
    .str.capitalize()
)

df["stay_days"] = pd.to_numeric(
    df["stay_days"],
    errors="coerce"
)

df["stay_days"] = (
    df["stay_days"]
    .fillna(1)
    .astype(int)
)

df.loc[
    df["stay_days"] < 1,
    "stay_days"
] = 1


# ============================================================
# REMOVE EMPTY DOCUMENTS
# ============================================================

df = df[
    df["document"].str.strip() != ""
].reset_index(drop=True)


if len(df) < 2:

    raise ValueError(
        "medical_documents.csv must contain "
        "at least 2 valid documents."
    )


print("\nClean dataset:")
print(df)


# ============================================================
# TEXT TOKENIZATION
# ============================================================

print("\nPreparing tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    df["document"]
)

sequences = tokenizer.texts_to_sequences(
    df["document"]
)

X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# ENCODE WARD TYPE
# ============================================================

ward_types = sorted(
    df["ward_type"].unique()
)

ward_to_id = {
    ward: index
    for index, ward in enumerate(ward_types)
}

df["ward_id"] = df[
    "ward_type"
].map(ward_to_id)


print("\nWard classes:")
print(ward_to_id)


# ============================================================
# ENCODE YES / NO
# ============================================================

def yes_no_to_int(value):

    value = str(value).strip().lower()

    if value in [
        "yes",
        "true",
        "1"
    ]:
        return 1

    return 0


y_oxygen = df[
    "oxygen"
].apply(
    yes_no_to_int
).values.astype(np.float32)

y_ventilator = df[
    "ventilator"
].apply(
    yes_no_to_int
).values.astype(np.float32)

y_isolation = df[
    "isolation"
].apply(
    yes_no_to_int
).values.astype(np.float32)


# ============================================================
# STAY DAYS
# ============================================================

y_stay = df[
    "stay_days"
].values.astype(np.float32)


# ============================================================
# WARD LABELS
# ============================================================

y_ward = tf.keras.utils.to_categorical(
    df["ward_id"],
    num_classes=len(ward_types)
)


# ============================================================
# BUILD BiLSTM MODEL
# ============================================================

print("\nBuilding BiLSTM model...")

input_layer = Input(
    shape=(MAX_SEQUENCE_LENGTH,),
    name="document_input"
)

embedding = Embedding(
    input_dim=MAX_WORDS,
    output_dim=64,
    name="word_embedding"
)(input_layer)

bilstm = Bidirectional(
    LSTM(
        64,
        return_sequences=False
    ),
    name="bilstm"
)(embedding)

dropout = Dropout(
    0.3
)(bilstm)


# ============================================================
# OUTPUT 1: WARD
# ============================================================

ward_output = Dense(
    len(ward_types),
    activation="softmax",
    name="ward_output"
)(dropout)


# ============================================================
# OUTPUT 2: OXYGEN
# ============================================================

oxygen_output = Dense(
    1,
    activation="sigmoid",
    name="oxygen_output"
)(dropout)


# ============================================================
# OUTPUT 3: VENTILATOR
# ============================================================

ventilator_output = Dense(
    1,
    activation="sigmoid",
    name="ventilator_output"
)(dropout)


# ============================================================
# OUTPUT 4: ISOLATION
# ============================================================

isolation_output = Dense(
    1,
    activation="sigmoid",
    name="isolation_output"
)(dropout)


# ============================================================
# OUTPUT 5: STAY DAYS
# ============================================================

stay_output = Dense(
    1,
    activation="linear",
    name="stay_output"
)(dropout)


# ============================================================
# COMPLETE MODEL
# ============================================================

model = Model(
    inputs=input_layer,
    outputs=[
        ward_output,
        oxygen_output,
        ventilator_output,
        isolation_output,
        stay_output
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss={
        "ward_output":
            "categorical_crossentropy",

        "oxygen_output":
            "binary_crossentropy",

        "ventilator_output":
            "binary_crossentropy",

        "isolation_output":
            "binary_crossentropy",

        "stay_output":
            "mse"
    },

    metrics={
        "ward_output":
            ["accuracy"],

        "oxygen_output":
            ["accuracy"],

        "ventilator_output":
            ["accuracy"],

        "isolation_output":
            ["accuracy"],

        "stay_output":
            ["mae"]
    }
)


print("\nModel summary:\n")

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining BiLSTM model...\n")

history = model.fit(

    X,

    {
        "ward_output":
            y_ward,

        "oxygen_output":
            y_oxygen,

        "ventilator_output":
            y_ventilator,

        "isolation_output":
            y_isolation,

        "stay_output":
            y_stay
    },

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.2,

    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving BiLSTM model...")

model.save(
    MODEL_FILE
)


# ============================================================
# SAVE TOKENIZER
# ============================================================

with open(
    TOKENIZER_FILE,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


# ============================================================
# SAVE LABEL INFORMATION
# ============================================================

label_information = {

    "ward_types":
        ward_types,

    "ward_to_id":
        ward_to_id,

    "max_words":
        MAX_WORDS,

    "max_sequence_length":
        MAX_SEQUENCE_LENGTH
}


with open(
    LABEL_ENCODER_FILE,
    "wb"
) as file:

    pickle.dump(
        label_information,
        file
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print(
    "BiLSTM NLP MODEL TRAINING COMPLETED"
)

print("=" * 60)

print(
    f"\nModel saved to:\n{MODEL_FILE}"
)

print(
    f"\nTokenizer saved to:\n{TOKENIZER_FILE}"
)

print(
    f"\nLabel information saved to:\n"
    f"{LABEL_ENCODER_FILE}"
)

print("\nThe model predicts:")

print("1. Ward type")
print("2. Oxygen requirement")
print("3. Ventilator requirement")
print("4. Isolation requirement")
print("5. Expected stay duration")

print("\nYou can now proceed to the next step.")