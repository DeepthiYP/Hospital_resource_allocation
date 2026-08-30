import os
import requests

from flask import Flask, render_template, request, jsonify

from services.document_processor import extract_text
from services.ml_model import predict_requirements

from services.occupancy import (
    get_occupancy_summary,
    get_occupied_beds,
    get_ward_summary
)


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FASTAPI_URL = "http://127.0.0.1:8000"


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def home():

    summary = get_occupancy_summary()
    occupied = get_occupied_beds()
    wards = get_ward_summary()

    return render_template(
        "index.html",
        summary=summary,
        occupied=occupied.to_dict(orient="records"),
        wards=wards.to_dict(orient="records")
    )


# ==========================================
# DOCUMENT ANALYSIS
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze_document():

    if "document" not in request.files:
        return jsonify({
            "error": "No document uploaded."
        }), 400

    file = request.files["document"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    try:

        # ==================================
        # 1. EXTRACT PDF TEXT
        # ==================================

        text = extract_text(file_path)

        if not text:
            return jsonify({
                "error": "Could not extract text from document."
            }), 400

        # ==================================
        # 2. ANALYSE DOCUMENT
        # ==================================

        requirements = predict_requirements(text)

        # ==================================
        # 3. PATIENT ID
        # ==================================

        patient_id = request.form.get(
            "patient_id",
            requirements.get("patient_id", "P999")
        )

        requirements["patient_id"] = patient_id

        # ==================================
        # DEBUG
        # ==================================

        print("\n==============================")
        print("DOCUMENT TEXT")
        print("==============================")
        print(text)

        print("\n==============================")
        print("EXTRACTED REQUIREMENTS")
        print("==============================")
        print(requirements)

        print("==============================\n")

        # ==================================
        # 4. CALL FASTAPI
        # ==================================

        payload = {
            "ward_type": requirements["ward_type"],
            "stay_days": requirements["stay_days"],
            "oxygen": requirements["oxygen"],
            "ventilator": requirements["ventilator"],
            "isolation": requirements["isolation"],
            "patient_id": patient_id
        }

        response = requests.post(
            FASTAPI_URL + "/allocate",
            json=payload,
            timeout=30
        )

        # ==================================
        # 5. FASTAPI ERROR
        # ==================================

        if response.status_code != 200:

            return jsonify({
                "error": "FastAPI allocation failed.",
                "details": response.text
            }), 500

        allocation = response.json()

        # ==================================
        # 6. FINAL RESPONSE
        # ==================================

        return jsonify({
            "document_text": text,
            "requirements": requirements,
            "allocation": allocation
        })

    except requests.exceptions.ConnectionError:

        return jsonify({
            "error": (
                "FastAPI server is not running. "
                "Start FastAPI using uvicorn."
            )
        }), 500

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )