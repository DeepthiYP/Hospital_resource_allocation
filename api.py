from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from services.allocator import allocate_bed
from services.forecasting_model import forecast_occupancy


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Hospital Resource Allocation API",
    description="Hospital bed allocation and occupancy forecasting API",
    version="1.0.0"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class AllocationRequest(BaseModel):

    ward_type: str

    stay_days: int

    oxygen: bool = False

    ventilator: bool = False

    isolation: bool = False

    patient_id: Optional[str] = "P999"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Hospital Resource Allocation API",
        "version": "1.0.0"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# BED ALLOCATION
# =========================================================

@app.post("/allocate")
def allocate(
    requirements: AllocationRequest
):

    data = requirements.dict()

    result = allocate_bed(
        data
    )

    return result


# =========================================================
# OCCUPANCY FORECAST
# =========================================================

@app.get("/forecast")
def forecast(
    days: int = 7
):

    if days < 1:

        days = 1

    if days > 30:

        days = 30

    try:

        result = forecast_occupancy(
            days
        )

        return {

            "status":
                "success",

            "forecast_days":
                days,

            "forecast":
                result

        }

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)

        }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )