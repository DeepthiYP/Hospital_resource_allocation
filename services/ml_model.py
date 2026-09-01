import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_FILE = "models/bilstm_nlp_model.keras"

TOKENIZER_FILE = "models/nlp_tokenizer.pkl"

LABEL_INFO_FILE = "models/nlp_label_info.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = None
tokenizer = None
label_info = None


def load_model():

    global model
    global tokenizer
    global label_info

    if model is None:

        if not os.path.exists(MODEL_FILE):

            raise FileNotFoundError(
                f"BiLSTM model not found: {MODEL_FILE}\n"
                "Run train_nlp_model.py first."
            )

        model = tf.keras.models.load_model(
            MODEL_FILE
        )

    if tokenizer is None:

        if not os.path.exists(TOKENIZER_FILE):

            raise FileNotFoundError(
                f"Tokenizer not found: {TOKENIZER_FILE}"
            )

        with open(
            TOKENIZER_FILE,
            "rb"
        ) as file:

            tokenizer = pickle.load(file)

    if label_info is None:

        if not os.path.exists(LABEL_INFO_FILE):

            raise FileNotFoundError(
                f"Label information not found: "
                f"{LABEL_INFO_FILE}"
            )

        with open(
            LABEL_INFO_FILE,
            "rb"
        ) as file:

            label_info = pickle.load(file)

    return model, tokenizer, label_info


# ============================================================
# YES / NO CONVERSION
# ============================================================

def yes_no(value):

    """
    Converts model probability into
    Yes / No.
    """

    if isinstance(value, (list, np.ndarray)):

        value = float(
            np.asarray(value).flatten()[0]
        )

    else:

        value = float(value)

    return "Yes" if value >= 0.5 else "No"


# ============================================================
# PREDICT PATIENT REQUIREMENTS
# ============================================================

def predict_requirements(text):

    if not text or not str(text).strip():

        raise ValueError(
            "Document text is empty."
        )

    model, tokenizer, label_info = (
        load_model()
    )

    max_sequence_length = label_info[
        "max_sequence_length"
    ]

    ward_types = label_info[
        "ward_types"
    ]

    # --------------------------------------------------------
    # Convert document to sequence
    # --------------------------------------------------------

    sequence = tokenizer.texts_to_sequences(
        [str(text)]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # BiLSTM prediction
    # --------------------------------------------------------

    predictions = model.predict(
        padded,
        verbose=0
    )

    (
        ward_prediction,
        oxygen_prediction,
        ventilator_prediction,
        isolation_prediction,
        stay_prediction
    ) = predictions

    # --------------------------------------------------------
    # Ward
    # --------------------------------------------------------

    ward_index = int(
        np.argmax(
            ward_prediction[0]
        )
    )

    if ward_index >= len(ward_types):

        ward_index = 0

    predicted_ward = ward_types[
        ward_index
    ]

    # --------------------------------------------------------
    # Oxygen
    # --------------------------------------------------------

    oxygen = yes_no(
        oxygen_prediction[0]
    )

    # --------------------------------------------------------
    # Ventilator
    # --------------------------------------------------------

    ventilator = yes_no(
        ventilator_prediction[0]
    )

    # --------------------------------------------------------
    # Isolation
    # --------------------------------------------------------

    isolation = yes_no(
        isolation_prediction[0]
    )

    # --------------------------------------------------------
    # Stay duration
    # --------------------------------------------------------

    stay_days = float(
        stay_prediction[0][0]
    )

    stay_days = max(
        1,
        round(stay_days)
    )

    # Limit unrealistic values
    stay_days = min(
        stay_days,
        60
    )

    # --------------------------------------------------------
    # Return requirements
    # --------------------------------------------------------

    requirements = {

        "ward_type":
            predicted_ward,

        "oxygen":
            oxygen == "Yes",

        "ventilator":
            ventilator == "Yes",

        "isolation":
            isolation == "Yes",

        "stay_days":
            int(stay_days)
    }

    return requirements


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "TESTING BiLSTM NLP MODEL"
    )

    print("=" * 60)

    test_document = (
        "Patient requires ICU admission "
        "with oxygen and ventilator support "
        "for 5 days."
    )

    result = predict_requirements(
        test_document
    )

    print("\nDocument:")
    print(test_document)

    print("\nPredicted requirements:")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )