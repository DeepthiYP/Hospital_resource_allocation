import os
import requests

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from services.document_processor import (
    extract_text
)

from services.ml_model import (
    predict_requirements
)

from services.occupancy import (
    get_occupancy_summary,
    get_occupied_beds,
    get_ward_summary
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# FASTAPI URL
# =========================================================

FASTAPI_URL = (
    "http://127.0.0.1:8000"
)


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def home():

    try:

        summary = (
            get_occupancy_summary()
        )

        occupied = (
            get_occupied_beds()
        )

        wards = (
            get_ward_summary()
        )

        return render_template(

            "index.html",

            summary=summary,

            occupied=occupied.to_dict(
                orient="records"
            ),

            wards=wards.to_dict(
                orient="records"
            )

        )

    except Exception as e:

        return render_template(

            "index.html",

            summary={},

            occupied=[],

            wards=[],

            error=str(e)

        )


# =========================================================
# DOCUMENT ANALYSIS + ALLOCATION
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze_document():

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "document" not in request.files:

        return jsonify({

            "error":
                "No document uploaded."

        }), 400


    file = request.files[
        "document"
    ]


    if file.filename == "":

        return jsonify({

            "error":
                "No file selected."

        }), 400


    # -----------------------------------------------------
    # SAVE DOCUMENT
    # -----------------------------------------------------

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        safe_filename

    )

    file.save(
        file_path
    )


    try:

        # =================================================
        # STEP 1 — EXTRACT DOCUMENT TEXT
        # =================================================

        text = extract_text(
            file_path
        )


        if not text:

            return jsonify({

                "error":
                    "Could not extract text from document."

            }), 400


        # =================================================
        # STEP 2 — BiLSTM NLP MODEL
        # =================================================

        requirements = (
            predict_requirements(text)
        )


        # =================================================
        # STEP 3 — PATIENT ID
        # =================================================

        patient_id = request.form.get(

            "patient_id",

            "P999"

        )


        requirements[
            "patient_id"
        ] = patient_id


        # =================================================
        # STEP 4 — SEND TO FASTAPI
        # =================================================

        response = requests.post(

            FASTAPI_URL
            + "/allocate",

            json={

                "ward_type":
                    requirements[
                        "ward_type"
                    ],

                "stay_days":
                    int(
                        requirements[
                            "stay_days"
                        ]
                    ),

                "oxygen":
                    bool(
                        requirements[
                            "oxygen"
                        ]
                    ),

                "ventilator":
                    bool(
                        requirements[
                            "ventilator"
                        ]
                    ),

                "isolation":
                    bool(
                        requirements[
                            "isolation"
                        ]
                    ),

                "patient_id":
                    patient_id

            },

            timeout=60

        )


        # =================================================
        # STEP 5 — FASTAPI ERROR
        # =================================================

        if response.status_code != 200:

            return jsonify({

                "error":
                    "FastAPI allocation failed.",

                "details":
                    response.text

            }), 500


        # =================================================
        # STEP 6 — ALLOCATION RESULT
        # =================================================

        allocation = (
            response.json()
        )


        # =================================================
        # STEP 7 — RETURN EVERYTHING
        # =================================================

        return jsonify({

            "document_text":
                text,

            "requirements":
                requirements,

            "allocation":
                allocation

        })


    # =====================================================
    # FASTAPI NOT RUNNING
    # =====================================================

    except requests.exceptions.ConnectionError:

        return jsonify({

            "error":
                (
                    "FastAPI server is not running. "
                    "Start it using: "
                    "python api.py"
                )

        }), 500


    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# OCCUPANCY FORECAST
# =========================================================

@app.route(
    "/forecast",
    methods=["GET"]
)
def forecast():

    try:

        days = request.args.get(
            "days",
            default=7,
            type=int
        )


        days = max(
            1,
            min(days, 30)
        )


        response = requests.get(

            FASTAPI_URL
            + "/forecast",

            params={
                "days": days
            },

            timeout=60

        )


        if response.status_code != 200:

            return jsonify({

                "error":
                    "Forecasting service failed.",

                "details":
                    response.text

            }), 500


        return jsonify(
            response.json()
        )


    except requests.exceptions.ConnectionError:

        return jsonify({

            "error":
                "FastAPI server is not running."

        }), 500


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# FASTAPI STATUS
# =========================================================

@app.route(
    "/api-status",
    methods=["GET"]
)
def api_status():

    try:

        response = requests.get(

            FASTAPI_URL
            + "/health",

            timeout=5

        )

        return jsonify({

            "online":
                response.status_code == 200

        })


    except Exception:

        return jsonify({

            "online":
                False

        })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )